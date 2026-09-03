import argparse
from typing import Any, Optional
from fastapi import WebSocket
import asyncio
from lc_conductor import ActionManager, handles
from lc_conductor.agents import AgentRecord, AgentRequest, AgentResponse
from lc_conductor import (
    BuiltinToolDefinition,
    ToolRuntime,
    resolve_builtin_tool_descriptors,
    resolve_allowed_backends,
    is_backend_allowed,
    is_custom_url_allowed,
)
from loguru import logger
from charge.tasks.task import Task
from charge_backend.backend_helper_funcs import (
    CallbackHandler,
    Reaction,
    FlaskRunSettings,
)
from charge_backend.flask_experiment import FlaskExperiment, GraphContext
from charge_backend.lmo.lmo_charge_backend_funcs import generate_lead_molecule
from charge_backend.charge_backend_custom import run_custom_problem
from functools import partial
from charge_backend.retrosynthesis.template import (
    template_based_retrosynthesis,
    compute_templates_for_node,
)
from charge_backend.retrosynthesis.ai import (
    ai_based_retrosynthesis,
    db_then_ai_retrosynthesis,
)
from charge_backend.retrosynthesis import route_planner
from charge_backend.retrosynthesis.template import run_ranked_retro_planner
from charge_backend.retrosynthesis.alternatives import set_reaction_alternative
from charge_backend.prompt_debugger import debug_prompt
from charge_backend.backend_helper_funcs import Node
from charge_backend.attachments import image_refs, validate_image_attachments
from charge_backend.pdf import PdfDocumentRegistry
from charge_backend.builtin_tools import list_builtin_tool_definitions


class FlaskActionManager(ActionManager):
    """Handles action state for a websocket connection."""

    def __init__(
        self,
        websocket: WebSocket,
        args: argparse.Namespace,
        username: str,
        pdf_registry: Optional[PdfDocumentRegistry] = None,
        builtin_tool_definitions: Optional[list[BuiltinToolDefinition]] = None,
    ):
        super().__init__(
            websocket,
            args,
            username,
            builtin_tool_definitions=builtin_tool_definitions,
        )
        self.experiment = FlaskExperiment(task=None)  # Use the flask experiment instead
        self.run_settings: FlaskRunSettings = FlaskRunSettings()
        self.retro_synth_context: Optional[GraphContext] = None
        self.pdf_registry = pdf_registry or PdfDocumentRegistry(self.agent_backend)
        self.builtin_tool_definitions = (
            builtin_tool_definitions
            or list_builtin_tool_definitions(
                self.pdf_registry,
                getattr(args, "config_file", None),
                experiment=self.experiment,
            )
        )

    async def cleanup(self):
        """
        Tears down the action manager
        """
        self.pdf_registry.cleanup()
        await super().cleanup()

    def get_retro_synth_context(self) -> GraphContext:
        return self.experiment.graph_context

    def setup_run_settings(self, data: dict[str, Any]):
        if "runSettings" in data:
            self.run_settings = FlaskRunSettings(**data["runSettings"])

    def reset_problem_context(self, problem_type: Optional[str]) -> None:
        """Reset backend-only state that is scoped to the active problem type."""
        self.experiment.reset()

    def selected_tool_runtime(self) -> ToolRuntime:
        runtime = super().selected_tool_runtime()
        if not self.pdf_registry.has_active_document():
            return runtime
        if any(tool.identifier == "consult_with_document" for tool in runtime.tools):
            return runtime
        return ToolRuntime(
            bearer_token=runtime.bearer_token,
            tools=[
                *runtime.tools,
                *resolve_builtin_tool_descriptors(
                    ["consult_with_document"],
                    self.builtin_tool_definitions,
                ),
            ],
        )

    def route_planner_tool_runtime(self) -> ToolRuntime:
        runtime = self.selected_tool_runtime()
        return ToolRuntime(
            bearer_token=runtime.bearer_token,
            tools=[
                *runtime.tools,
                *resolve_builtin_tool_descriptors(
                    ["verify_smiles", "canonicalize_smiles"],
                    self.builtin_tool_definitions,
                ),
            ],
        )

    def _agent_update_callback(self, agent_key: str, data: dict[str, Any]):
        return partial(self.send_agent_update, agent_key, debug=bool(data.get("debug")))

    def _route_planning_callback(self, agent_key: str, data: dict[str, Any]):
        return CallbackHandler(
            self.websocket,
            agent_key=agent_key,
            on_agent_update=self._agent_update_callback(agent_key, data),
        )

    def _document_reference_context(self) -> str:
        metadata = self.pdf_registry.active_metadata()
        if metadata is None:
            return ""
        return (
            "\n\nAn uploaded PDF reference is available through the "
            "`consult_with_document` tool. Use it when the user asks about the "
            f"reference document `{metadata.title}` ({metadata.name}); cite page "
            "numbers from the tool output."
        )

    def _with_document_reference_context(self, system_prompt: str) -> str:
        return f"{system_prompt}{self._document_reference_context()}"

    @handles("configure-pdf-reference")
    async def handle_configure_pdf_reference(self, data: dict[str, Any]) -> None:
        reference = data.get("pdfReference")
        if reference is None:
            self.pdf_registry.clear()
            if data.get("silent"):
                return
            await self.websocket.send_json(
                {
                    "type": "pdf-reference-response",
                    "reference": None,
                }
            )
            return

        try:
            metadata = await asyncio.to_thread(
                self.pdf_registry.set_from_attachment,
                reference,
            )
        except ValueError as exc:
            await self.websocket.send_json(
                {
                    "type": "pdf-reference-response",
                    "reference": {
                        "id": (
                            str(reference.get("id") or "pdf_reference")
                            if isinstance(reference, dict)
                            else "pdf_reference"
                        ),
                        "name": (
                            str(reference.get("name") or "Reference PDF")
                            if isinstance(reference, dict)
                            else "Reference PDF"
                        ),
                        "mimeType": "application/pdf",
                        "sizeBytes": (
                            int(reference.get("sizeBytes") or 0)
                            if isinstance(reference, dict)
                            else 0
                        ),
                        "status": "error",
                        "error": str(exc),
                    },
                }
            )
            return
        await self.websocket.send_json(
            {
                "type": "pdf-reference-response",
                "reference": metadata.json(),
            }
        )

    @handles("compute")
    async def handle_compute(self, data: dict) -> None:
        self.setup_run_settings(data)
        problem_type = data.get("problemType")
        self.reset_problem_context(problem_type)
        if problem_type == "optimization":
            asyncio.create_task(self._handle_optimization(data))
        elif problem_type == "retrosynthesis":
            asyncio.create_task(self._handle_retrosynthesis(data))
        elif problem_type == "route-planning":
            asyncio.create_task(self._handle_route_planning(data))
        elif problem_type == "custom":
            asyncio.create_task(self._handle_custom_problem(data))
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")

    async def _handle_optimization(
        self,
        data: dict,
        initial_level: int = 0,
        initial_node_id: int = 0,
        initial_x_position: int = 50,
    ) -> None:
        """Handle optimization problem type."""
        await self.task_manager.clogger.info("Start Optimization action received")
        logger.info(f"Data: {data}")

        # Validate required field
        if "propertyType" not in data:
            error_msg = "Missing required field 'propertyType' for optimization problem"
            logger.error(error_msg)
            await self.websocket.send_json(
                {
                    "type": "response",
                    "message": {
                        "source": "System",
                        "message": f"Error: {error_msg}",
                    },
                }
            )
            await self.websocket.send_json({"type": "complete"})
            return

        property_type = data["propertyType"]

        # Property attributes for prompting
        if property_type == "custom":
            # Validate custom property fields
            if not data.get("customPropertyName") or not data.get("customPropertyDesc"):
                error_msg = "Custom property requires both 'customPropertyName' and 'customPropertyDesc'"
                logger.error(error_msg)
                await self.websocket.send_json(
                    {
                        "type": "response",
                        "message": {
                            "source": "System",
                            "message": f"Error: {error_msg}",
                        },
                    }
                )
                await self.websocket.send_json({"type": "complete"})
                return

            property_attributes = (
                data["customPropertyName"],
                data["customPropertyDesc"],
                "calculate_property_hf",
                "greater" if data["customPropertyAscending"] else "less",
            )
        else:
            property_name = property_type
            DEFAULT_PROPERTIES = {
                "density": (
                    "density",
                    "crystalline density (g/cm^3)",
                    "calculate_property_hf",
                    "greater",
                ),
                "hof": (
                    "heat of formation",
                    "Heat of formation (kcal/mol)",
                    "calculate_property_hf",
                    "greater",
                ),
                "bandgap": (
                    "band gap",
                    "HOMO-LUMO energy gap (Hartree)",
                    "calculate_property_hf",
                    "greater",
                ),
            }
            if property_name not in DEFAULT_PROPERTIES:
                error_msg = (
                    f"Property '{property_name}' not found in default properties"
                )
                logger.error(error_msg)
                await self.websocket.send_json(
                    {
                        "type": "response",
                        "message": {
                            "source": "System",
                            "message": f"Error: {error_msg}",
                        },
                    }
                )
                await self.websocket.send_json({"type": "complete"})
                return
            property_attributes = DEFAULT_PROPERTIES[property_name]

        # Extract customization parameters
        customization = data.get("customization", {})
        enable_constraints = customization.get("enableConstraints", False)
        molecular_similarity = customization.get("molecularSimilarity", 0.7)
        diversity_penalty = customization.get("diversityPenalty", 0.0)
        exploration_rate = customization.get("explorationRate", 0.5)
        additional_constraints = customization.get("additionalConstraints", [])
        number_of_molecules = customization.get("numberOfMolecules", 10)
        num_top_candidates = customization.get("numTopCandidates", 3)
        depth = customization.get("depth", 3)
        tool_runtime = self.selected_tool_runtime()
        attachments = validate_image_attachments(data)
        if not data.get("_agentRequestAnnounced"):
            request_text = (
                f"LMO request: {data.get('query')}"
                if data.get("query")
                else f"LMO request for {data.get('smiles', 'current molecule')}"
            )
            await self._send_processing_message(
                request_text,
                source="LMO request",
                images=image_refs(attachments) if attachments else None,
            )

        run_func = partial(
            generate_lead_molecule,
            data["smiles"],
            self.experiment,
            self.args.json_file,
            self.args.max_retries,
            depth,
            tool_runtime,
            self.task_manager.websocket,
            self.run_settings,
            *property_attributes,
            data.get("query", None),
            initial_level,
            initial_node_id,
            initial_x_position,
            enable_constraints,
            molecular_similarity,
            diversity_penalty,
            exploration_rate,
            additional_constraints,
            number_of_molecules,
            num_top_candidates,
            attachments,
            self._agent_update_callback("lmo:main", data),
        )
        await self.task_manager.run_task(run_func())

    async def _handle_retrosynthesis(self, data: dict) -> None:
        """Handle retrosynthesis problem type."""
        run_func = partial(
            template_based_retrosynthesis,
            data["smiles"],
            self.args.config_file,
            self.get_retro_synth_context(),
            self.task_manager.websocket,
            self.run_settings,
            data.get("RetroMode", "single"),
            self.experiment,
        )

        await self.task_manager.run_task(run_func())

    async def _handle_route_planning(self, data: dict) -> None:
        """Draft route-level retrosynthesis plans for review."""
        attachments = validate_image_attachments(data)
        smiles = data["smiles"]

        await self._send_processing_message(
            f"Planning candidate routes for {smiles}",
            source="Route Planner",
            images=image_refs(attachments) if attachments else None,
        )

        async def run_func() -> None:
            summarizer_callback = self._route_planning_callback(
                "route-planning:summarizer",
                data,
            )
            planner_callback = self._route_planning_callback(
                "route-planning:planner",
                data,
            )
            clogger = self.task_manager.clogger
            await self._send_processing_message(
                f"Enumerating template routes for {smiles}.",
                source="Route Planner",
                eventKind="status",
            )
            reaction, routes, selected_candidates = await run_ranked_retro_planner(
                self.args.config_file,
                smiles,
                clogger,
                self.run_settings,
            )
            del reaction, routes
            await self._send_processing_message(
                "Template route search completed: "
                f"{len(selected_candidates)} ranked routes selected.",
                source="Route Planner",
                eventKind="status",
            )

            if not selected_candidates:
                await self._send_processing_message(
                    f"No ranked template routes found for {smiles}.",
                    source="Route Planner",
                )
                await self.websocket.send_json({"type": "complete"})
                return

            summaries = await route_planner.summarize_routes(
                selected_candidates,
                self.experiment,
                limit=10,
                concurrency=10,
                callback=summarizer_callback,
                status_callback=partial(
                    self._send_processing_message,
                    source="Route Planner",
                    agentKey="route-planning:summarizer",
                    eventKind="status",
                ),
            )
            await summarizer_callback.drain()
            route_context = route_planner.build_route_context(summaries, limit=10)

            await self._send_processing_message(
                "Planner started generating candidate plans.",
                source="Route Planner",
                agentKey="route-planning:planner",
                eventKind="status",
            )

            async def report_planner_progress() -> None:
                elapsed = 0
                while True:
                    await asyncio.sleep(30)
                    elapsed += 30
                    await self._send_processing_message(
                        "Planner is still generating candidate plans "
                        f"({elapsed} seconds elapsed).",
                        source="Route Planner",
                        agentKey="route-planning:planner",
                        eventKind="status",
                    )

            progress_task = asyncio.create_task(report_planner_progress())
            try:
                output = await route_planner.plan_candidate_routes(
                    smiles,
                    route_context,
                    self.experiment,
                    self.route_planner_tool_runtime(),
                    user_request=data.get("query"),
                    attachments=attachments,
                    agent_key="route-planning:planner",
                    callback=planner_callback,
                )
            finally:
                progress_task.cancel()
                await asyncio.gather(progress_task, return_exceptions=True)

            await planner_callback.drain()
            await self._send_processing_message(
                "Planner completed candidate plan generation.",
                source="Route Planner",
                agentKey="route-planning:planner",
                eventKind="status",
            )
            route_planning_result = route_planner.route_planning_result_from_output(
                smiles,
                output,
                user_constraints=data.get("query"),
                route_context=route_context,
            )
            route_planning_result.base_branch_state = (
                route_planner.save_compact_route_planning_branch_state(
                    self.experiment,
                    route_planning_result,
                )
            )
            self.experiment.route_planning_result = route_planning_result
            await self._send_processing_message(
                route_planner.format_route_planning_output(output),
                source="Route Planner",
            )
            await self.websocket.send_json(
                {
                    "type": "route-planning-result-response",
                    "result": route_planner.route_planning_result_payload(
                        route_planning_result
                    ),
                }
            )
            await self.websocket.send_json(
                {
                    "type": "route-planning-response",
                    "plans": output.model_dump(mode="json"),
                }
            )
            await self.websocket.send_json({"type": "complete"})

        await self.task_manager.run_task(run_func())

    async def _run_in_route_plan_branch(
        self,
        result,
        plan_id: str,
        action,
        *,
        latest_user_message: str | None = None,
        latest_assistant_message=None,
    ):
        _, plan = route_planner.find_route_plan(result, plan_id)
        branch_state = plan.branch_state or result.base_branch_state
        route_planner.restore_route_planning_branch_state(
            self.experiment, branch_state, result
        )
        output = await action()
        _, current_plan = route_planner.find_route_plan(result, plan_id)
        assistant_message = (
            latest_assistant_message(output)
            if callable(latest_assistant_message)
            else latest_assistant_message
        )
        current_plan.branch_state = (
            route_planner.save_compact_route_planning_branch_state(
                self.experiment,
                result,
                latest_user_message=latest_user_message,
                latest_assistant_message=assistant_message,
            )
        )
        self.experiment.route_planning_result = result
        return output

    def _route_planning_planner_record(
        self,
        result,
        plan_id: str | None = None,
    ) -> AgentRecord | None:
        branch_state = result.base_branch_state
        if plan_id:
            try:
                _, plan = route_planner.find_route_plan(result, plan_id)
            except ValueError:
                plan = None
            if plan is not None:
                branch_state = plan.branch_state or result.base_branch_state

        agent_sessions = (
            branch_state.get("agentSessions", {})
            if isinstance(branch_state, dict)
            else {}
        )
        record = agent_sessions.get("route-planning:planner")
        return AgentRecord.model_validate(record) if record is not None else None

    async def _send_route_planning_agent_update(
        self,
        chat_agent_key: str | None,
        result,
        plan_id: str | None = None,
    ) -> None:
        if not chat_agent_key:
            return
        record = self._route_planning_planner_record(result, plan_id)
        if record is None:
            return
        await self.websocket.send_json(
            AgentResponse(agentKey=chat_agent_key, agent=record).model_dump(
                exclude_none=True
            )
        )

    @handles("route-planning-question")
    async def handle_route_planning_question(self, data: dict) -> None:
        result = self.experiment.route_planning_result
        if result is None:
            await self._send_processing_message(
                "No route-planning result is available.",
                source="Route Planner",
            )
            await self.websocket.send_json({"type": "complete"})
            return

        async def run_func() -> None:
            plan_id = str(data["planId"])
            planner_callback = self._route_planning_callback(
                "route-planning:planner",
                data,
            )
            answer = await self._run_in_route_plan_branch(
                result,
                plan_id,
                lambda: route_planner.answer_route_plan_question(
                    result,
                    plan_id,
                    str(data.get("query") or ""),
                    self.experiment,
                    self.route_planner_tool_runtime(),
                    callback=planner_callback,
                ),
                latest_user_message=str(data.get("query") or ""),
                latest_assistant_message=lambda answer: answer,
            )
            await planner_callback.drain()
            await self.websocket.send_json(
                {
                    "type": "route-planning-question-response",
                    "planId": plan_id,
                    "answer": answer,
                }
            )
            await self._send_route_planning_agent_update(
                data.get("chatAgentKey"),
                result,
                plan_id,
            )
            await self.websocket.send_json({"type": "complete"})

        await self.task_manager.run_task(run_func())

    @handles("route-planning-refine")
    async def handle_route_planning_refine(self, data: dict) -> None:
        result = self.experiment.route_planning_result
        if result is None:
            await self._send_processing_message(
                "No route-planning result is available.",
                source="Route Planner",
            )
            await self.websocket.send_json({"type": "complete"})
            return

        async def run_func() -> None:
            plan_id = str(data["planId"])
            planner_callback = self._route_planning_callback(
                "route-planning:planner",
                data,
            )
            await self._run_in_route_plan_branch(
                result,
                plan_id,
                lambda: route_planner.refine_route_plan_from_user_guidance(
                    result,
                    plan_id,
                    str(data.get("query") or ""),
                    self.experiment,
                    self.route_planner_tool_runtime(),
                    callback=planner_callback,
                ),
                latest_user_message=str(data.get("query") or ""),
                latest_assistant_message=(
                    "Route updated. Review the revised plan in the route planner."
                ),
            )
            await planner_callback.drain()
            if data.get("rematerialize"):
                graph_context = route_planner.commit_route_plan_to_graph(
                    result,
                    plan_id,
                    self.experiment,
                    self.run_settings.molecule_name_format,
                )
                await self.websocket.send_json(
                    {
                        "type": "route-planning-select-response",
                        "planId": plan_id,
                        "graphContext": graph_context.save_state(),
                        "result": route_planner.route_planning_result_payload(result),
                    }
                )
            else:
                await self.websocket.send_json(
                    {
                        "type": "route-planning-result-response",
                        "result": route_planner.route_planning_result_payload(result),
                    }
                )
            await self._send_route_planning_agent_update(
                data.get("chatAgentKey"),
                result,
                plan_id,
            )
            await self.websocket.send_json({"type": "complete"})

        await self.task_manager.run_task(run_func())

    @handles("route-planning-more")
    async def handle_route_planning_more(self, data: dict) -> None:
        result = self.experiment.route_planning_result
        if result is None:
            await self._send_processing_message(
                "No route-planning result is available.",
                source="Route Planner",
            )
            await self.websocket.send_json({"type": "complete"})
            return

        async def run_func() -> None:
            planner_callback = self._route_planning_callback(
                "route-planning:planner",
                data,
            )
            route_planner.restore_route_planning_branch_state(
                self.experiment,
                result.base_branch_state,
                result,
            )
            output = await route_planner.plan_more_candidate_routes(
                result,
                self.experiment,
                self.route_planner_tool_runtime(),
                user_feedback=str(data.get("query") or ""),
                callback=planner_callback,
            )
            await planner_callback.drain()
            result.base_branch_state = (
                route_planner.save_compact_route_planning_branch_state(
                    self.experiment,
                    result,
                    latest_user_message=str(data.get("query") or ""),
                    latest_assistant_message=(
                        "Generated additional route plans."
                        if output is not None
                        else None
                    ),
                )
            )
            self.experiment.route_planning_result = result
            await self.websocket.send_json(
                {
                    "type": "route-planning-result-response",
                    "result": route_planner.route_planning_result_payload(result),
                }
            )
            await self._send_route_planning_agent_update(
                data.get("chatAgentKey"),
                result,
            )
            await self.websocket.send_json({"type": "complete"})

        await self.task_manager.run_task(run_func())

    @handles("route-planning-evaluate")
    async def handle_route_planning_evaluate(self, data: dict) -> None:
        result = self.experiment.route_planning_result
        if result is None:
            await self._send_processing_message(
                "No route-planning result is available.",
                source="Route Planner",
            )
            await self.websocket.send_json({"type": "complete"})
            return

        async def run_func() -> None:
            plan_id = str(data["planId"])
            evaluator_callback = self._route_planning_callback(
                "route-planning:evaluator",
                data,
            )
            decision = await self._run_in_route_plan_branch(
                result,
                plan_id,
                lambda: route_planner.evaluate_selected_route_plan(
                    result,
                    plan_id,
                    self.experiment,
                    self.route_planner_tool_runtime(),
                    evaluator_callback=evaluator_callback,
                    pipette_status_callback=partial(
                        self._send_processing_message,
                        source="Pipette",
                        agentKey="route-planning:evaluator",
                        eventKind="status",
                    ),
                    status_callback=partial(
                        self._send_processing_message,
                        source="Route Evaluator",
                        agentKey="route-planning:evaluator",
                        eventKind="status",
                    ),
                ),
            )
            await evaluator_callback.drain()
            await self.websocket.send_json(
                {
                    "type": "route-planning-evaluation-response",
                    "decision": route_planner.route_evaluation_decision_payload(
                        decision
                    ),
                }
            )
            await self.websocket.send_json({"type": "complete"})

        await self.task_manager.run_task(run_func())

    @handles("route-planning-apply-fixes")
    async def handle_route_planning_apply_fixes(self, data: dict) -> None:
        result = self.experiment.route_planning_result
        if result is None:
            await self._send_processing_message(
                "No route-planning result is available.",
                source="Route Planner",
            )
            await self.websocket.send_json({"type": "complete"})
            return

        async def run_func() -> None:
            plan_id = str(data["planId"])
            planner_callback = self._route_planning_callback(
                "route-planning:planner",
                data,
            )
            evaluator_callback = self._route_planning_callback(
                "route-planning:evaluator",
                data,
            )
            fix_options = [
                route_planner.RouteEvaluationFixOption.model_validate(option)
                for option in data.get("selectedFixOptions", [])
            ]
            decision = await self._run_in_route_plan_branch(
                result,
                plan_id,
                lambda: route_planner.apply_evaluator_fixes_to_route_plan(
                    result,
                    plan_id,
                    self.experiment,
                    self.route_planner_tool_runtime(),
                    user_guidance=str(data.get("query") or ""),
                    selected_fix_options=fix_options,
                    planner_callback=planner_callback,
                    evaluator_callback=evaluator_callback,
                    pipette_status_callback=partial(
                        self._send_processing_message,
                        source="Pipette",
                        agentKey="route-planning:evaluator",
                        eventKind="status",
                    ),
                    status_callback=partial(
                        self._send_processing_message,
                        source="Route Evaluator",
                        agentKey="route-planning:evaluator",
                        eventKind="status",
                    ),
                ),
                latest_user_message=str(data.get("query") or ""),
                latest_assistant_message=lambda decision: decision.message
                or "Evaluator fixes were applied and the revised route was evaluated.",
            )
            await planner_callback.drain()
            await evaluator_callback.drain()
            await self.websocket.send_json(
                {
                    "type": "route-planning-evaluation-response",
                    "decision": route_planner.route_evaluation_decision_payload(
                        decision
                    ),
                }
            )
            await self.websocket.send_json({"type": "complete"})

        await self.task_manager.run_task(run_func())

    @handles("route-planning-continue")
    async def handle_route_planning_continue(self, data: dict) -> None:
        result = self.experiment.route_planning_result
        if result is None:
            await self._send_processing_message(
                "No route-planning result is available.",
                source="Route Planner",
            )
            await self.websocket.send_json({"type": "complete"})
            return

        plan_id = str(data["planId"])
        decision = route_planner.continue_with_evaluated_route_plan(result, plan_id)
        await self.websocket.send_json(
            {
                "type": "route-planning-evaluation-response",
                "decision": route_planner.route_evaluation_decision_payload(decision),
            }
        )
        await self.websocket.send_json({"type": "complete"})

    @handles("route-planning-select")
    async def handle_route_planning_select(self, data: dict) -> None:
        result = self.experiment.route_planning_result
        if result is None:
            await self._send_processing_message(
                "No route-planning result is available.",
                source="Route Planner",
            )
            await self.websocket.send_json({"type": "complete"})
            return

        async def run_func() -> None:
            plan_id = str(data["planId"])

            async def commit_selected_plan():
                return route_planner.commit_route_plan_to_graph(
                    result,
                    plan_id,
                    self.experiment,
                    self.run_settings.molecule_name_format,
                )

            graph_context = await self._run_in_route_plan_branch(
                result,
                plan_id,
                commit_selected_plan,
            )
            await self.websocket.send_json(
                {
                    "type": "route-planning-select-response",
                    "planId": plan_id,
                    "graphContext": graph_context.save_state(),
                    "result": route_planner.route_planning_result_payload(result),
                }
            )
            await self.websocket.send_json({"type": "complete"})

        await self.task_manager.run_task(run_func())

    async def _handle_custom_problem(self, data: dict) -> None:
        """Handle custom problem type."""
        await self.task_manager.clogger.info("Setting up custom task...")
        logger.info(f"Data: {data}")
        tool_runtime = self.selected_tool_runtime()
        attachments = validate_image_attachments(data)
        await self._send_processing_message(
            data.get("userPrompt") or "Custom prompt",
            source="User",
            images=image_refs(attachments) if attachments else None,
        )

        run_func = partial(
            run_custom_problem,
            data["smiles"],
            self._with_document_reference_context(data["systemPrompt"]),
            data["userPrompt"],
            self.experiment,
            tool_runtime,
            self.task_manager.websocket,
            self.run_settings,
            attachments,
            self._agent_update_callback("custom:main", data),
        )

        await self.task_manager.run_task(run_func())

    @handles("compute-reaction-from")
    async def handle_compute_reaction_from(self, data: dict) -> None:
        """Handle compute-reaction-from action."""
        self.setup_run_settings(data)
        attachments = validate_image_attachments(data)

        logger.info("Synthesize tree leaf action received")
        logger.info(f"Data: {data}")
        node = self.experiment.graph_context.node_ids[data["nodeId"]]
        request_text = (
            f"Retrosynthesis request for node {data['nodeId']}: {data.get('query', '')}".strip()
            if data.get("query")
            else f"Retrosynthesis request for node {data['nodeId']}"
        )
        await self._send_processing_message(
            request_text,
            source="Retrosynthesis Request",
            images=image_refs(attachments) if attachments else None,
        )

        has_children = any(
            v == data["nodeId"] for v in self.experiment.graph_context.parents.values()
        )

        await self.experiment.graph_context.delete_subtree(
            data["nodeId"], self.websocket
        )
        await self.websocket.send_json(
            {
                "type": "node_update",
                "node": {
                    "id": data["nodeId"],
                    "reaction": Reaction(
                        "searching_0",
                        (
                            "Recomputing: searching database..."
                            if has_children
                            else "Searching database..."
                        ),
                        highlight="red",
                        label="Recomputing" if has_children else "Computing",
                    ).json(),
                },
            }
        )

        run_func = partial(
            (
                ai_based_retrosynthesis
                if data.get("aiOnly", False)
                else db_then_ai_retrosynthesis
            ),
            data["nodeId"],
            data.get("query", None),
            None,  # Unconstrained
            self.task_manager.websocket,
            self.experiment,
            self.args.config_file,
            self.run_settings,
            self.selected_tool_runtime(),
            attachments,
            self._agent_update_callback(f"reaction:{data['nodeId']}", data),
        )

        asyncio.create_task(self.task_manager.run_task(run_func()))

    @handles("compute-reaction-templates")
    async def handle_template_retrosynthesis(self, data: dict) -> None:
        """Handle compute-reaction-templates action."""
        self.setup_run_settings(data)

        logger.info("Synthesize tree leaf action received")
        logger.info(f"Data: {data}")
        node = self.experiment.graph_context.node_ids[data["nodeId"]]

        run_func = partial(
            compute_templates_for_node,
            node,
            self.args.config_file,
            self.experiment.graph_context,
            self.task_manager.websocket,
            self.run_settings,
        )

        asyncio.create_task(self.task_manager.run_task(run_func()))

    @handles("optimize-from")
    async def handle_optimize_from(self, data: dict) -> None:
        """Handle optimize-from action."""
        self.setup_run_settings(data)
        attachments = validate_image_attachments(data)
        prompt = data.get("query")
        if prompt:
            await self._send_processing_message(
                f"Processing optimization query: {prompt} for node {data['nodeId']}",
                source="LMO Request",
                images=image_refs(attachments) if attachments else None,
            )
        else:
            await self._send_processing_message(
                f"Processing optimization refinement for node {data['nodeId']}",
                source="LMO Request",
            )
        data["_agentRequestAnnounced"] = True

        node: str = data["nodeId"]
        if "_" in node:
            node = node[node.rfind("_") + 1 :]
        node_id = int(node)
        # Level is always equal to node ID for now.
        # TODO: Keep LMO context with nodes
        level = node_id
        xpos = data["xpos"]

        asyncio.create_task(
            self._handle_optimization(
                data,
                initial_level=level,
                initial_node_id=node_id,
                initial_x_position=xpos,
            )
        )

    @handles("recompute-parent-reaction")
    async def handle_recompute_reaction(self, data: dict) -> None:
        """Handle recompute-reaction action."""
        self.setup_run_settings(data)
        attachments = validate_image_attachments(data)

        # Get parent
        if data["nodeId"] not in self.experiment.graph_context.parents:
            await self._send_processing_message(
                f"Cannot find parent reaction for node {data['nodeId']}", source="Agent"
            )
            await self.websocket.send_json({"type": "complete"})
            return

        parent_nodeid = self.experiment.graph_context.parents[data["nodeId"]]
        parent_node = self.experiment.graph_context.node_ids[parent_nodeid]
        smiles = self.experiment.graph_context.node_ids[data["nodeId"]].smiles
        request_text = (
            f"Retrosynthesis request for node {data['nodeId']}: {data.get('query', '')}".strip()
            if data.get("query")
            else f"Retrosynthesis request for node {data['nodeId']}"
        )
        await self._send_processing_message(
            request_text,
            source="Retrosynthesis Request",
            images=image_refs(attachments) if attachments else None,
        )

        # Clear subtree and levels for layouting
        await self.experiment.graph_context.delete_subtree(
            parent_nodeid, self.websocket
        )
        await self.websocket.send_json(
            {
                "type": "node_update",
                "node": {
                    "id": parent_nodeid,
                    "reaction": Reaction(
                        "ai_reaction_0",
                        "Recomputing with AI orchestrator",
                        highlight="red",
                        label="Recomputing",
                    ).json(),
                },
            }
        )

        run_func = partial(
            ai_based_retrosynthesis,
            parent_nodeid,
            data.get("query", None),
            smiles,
            self.task_manager.websocket,
            self.experiment,
            self.args.config_file,
            self.run_settings,
            self.selected_tool_runtime(),
            attachments,
            self._agent_update_callback(f"reaction:{parent_nodeid}", data),
        )

        asyncio.create_task(self.task_manager.run_task(run_func()))

    @handles("set-reaction-alternative")
    async def handle_set_reaction_alternative(self, data: dict) -> None:
        """Handle set-reaction-alternative action."""
        self.setup_run_settings(data)

        # Get node
        if data["nodeId"] not in self.experiment.graph_context.node_ids:
            await self._send_processing_message(
                f"Cannot find node {data['nodeId']}", source="Agent"
            )
            await self.websocket.send_json({"type": "complete"})
            return
        node = self.experiment.graph_context.node_ids[data["nodeId"]]
        alt = data["alternativeId"]
        if not node.reaction or not node.reaction.alternatives:
            await self._send_processing_message(
                f"No alternative found for {data['nodeId']}", source="Agent"
            )
            await self.websocket.send_json({"type": "complete"})
            return

        await set_reaction_alternative(
            node, alt, self.experiment.graph_context, self.websocket
        )

    @handles("ui-update-orchestrator-settings")
    async def handle_orchestrator_settings_update(self, data: dict) -> None:
        if "moleculeName" in data:
            self.run_settings.molecule_name_format = data["moleculeName"]

        # Enforce the deployment backend allow-list server-side so a crafted
        # WebSocket message cannot bypass the UI restrictions.
        allowed = resolve_allowed_backends()
        if allowed:
            backend = data.get("backend")
            if not is_backend_allowed(backend, allowed):
                logger.warning(
                    f"Rejecting orchestrator update: backend '{backend}' is not "
                    f"in the allowed list {[e['backend'] for e in allowed]}"
                )
                await self.websocket.send_json(
                    {
                        "type": "response",
                        "message": {
                            "source": "System",
                            "message": (
                                f"Backend '{backend}' is not permitted by this "
                                "deployment. Settings were not applied."
                            ),
                        },
                    }
                )
                return

            # Drop a custom URL when this backend is locked to its preconfigured
            # endpoint, so the base handler resolves the default URL instead.
            if data.get("useCustomUrl") and not is_custom_url_allowed(backend, allowed):
                logger.warning(
                    f"Ignoring custom URL for backend '{backend}': custom URLs "
                    "are not permitted for this backend"
                )
                data["useCustomUrl"] = False

        await super().handle_orchestrator_settings_update(data)

    @handles("query-molecule")
    async def handle_custom_query_molecule(self, data: dict) -> None:
        """Handle a query on the given molecule."""
        self.setup_run_settings(data)
        attachments = validate_image_attachments(data)
        await self._send_processing_message(
            f"Processing molecule query: {data['query']} for node {data['nodeId']}",
            source="User",
            images=image_refs(attachments) if attachments else None,
        )
        smiles = data["smiles"]
        tool_runtime = self.selected_tool_runtime()

        task = Task(
            system_prompt=self._with_document_reference_context(
                f"You are a helpful chemical assistant who answers in concise but factual responses. Answer the following query about the molecule given by the SMILES string `{smiles}`."
            ),
            user_prompt=data["query"],
            attachments=attachments,
            **tool_runtime.task_kwargs(),
        )

        # Use the full experiment state
        callback_handler = CallbackHandler(
            self.websocket,
            agent_key=f"molecule:{data['nodeId']}",
            on_agent_update=self._agent_update_callback(
                f"molecule:{data['nodeId']}", data
            ),
        )
        agent = self.experiment.create_agent_with_experiment_state(
            task=task,
            agent_key=f"molecule:{data['nodeId']}",
            callback=callback_handler,
        )

        async def run_and_report():
            if self.run_settings.prompt_debugging:
                await debug_prompt(agent, self.websocket)
            result = await agent.run()
            await callback_handler.drain()
            self.experiment.add_to_context(agent, task, result)
            # Report answer
            await self._send_processing_message(
                result,
                source="Agent",
            )
            await self.websocket.send_json({"type": "complete"})

        asyncio.create_task(self.task_manager.run_task(run_and_report()))

    @handles("query-reaction")
    async def handle_custom_query_reaction(self, data: dict) -> None:
        """Handle a query on the reaction (from nodeId to its reactants)."""
        self.setup_run_settings(data)
        node_id = str(data["nodeId"])
        attachments = validate_image_attachments(data)

        await self._send_processing_message(
            f"Processing reaction query: {data['query']} for node {node_id}",
            source="User",
            images=image_refs(attachments) if attachments else None,
        )

        tool_runtime = self.selected_tool_runtime()

        task = Task(
            system_prompt="",
            user_prompt=data["query"],
            attachments=attachments,
            **tool_runtime.task_kwargs(),
        )

        reaction_str = self._reaction_context_for_node(node_id, data)

        task.system_prompt = self._with_document_reference_context(
            "You are a helpful chemical assistant who answers in concise but factual "
            "responses. Given the following reaction (as SMILES strings):\n"
            f"{reaction_str}\n\nAnswer the following query."
        )

        callback_handler = CallbackHandler(
            self.websocket,
            agent_key=f"reaction:{node_id}",
            on_agent_update=self._agent_update_callback(f"reaction:{node_id}", data),
        )
        agent = self.experiment.create_agent_with_experiment_state(
            task=task,
            agent_key=f"reaction:{node_id}",
            callback=callback_handler,
        )

        # TODO(later): For some reason the below code does not work because memory is not maintained
        # else:
        #     task.system_prompt = "You are a helpful chemical assistant who answers in concise but factual responses."
        #     task.user_prompt = (
        #         "Given the last computed reaction, answer the following query."
        #     )
        #     agent = self.retro_synth_context.node_id_to_charge_client[data["nodeId"]]
        #     agent.task = task

        async def run_and_report():
            if self.run_settings.prompt_debugging:
                await debug_prompt(agent, self.websocket)
            result = await agent.run()
            await callback_handler.drain()
            self.experiment.add_to_context(agent, task, result)
            # Report answer
            await self._send_processing_message(
                result,
                source="Agent",
            )
            await self.websocket.send_json({"type": "complete"})

        asyncio.create_task(self.task_manager.run_task(run_and_report()))

    def _reaction_hover_info_for_node(
        self, node_id: str, data: Optional[dict[str, Any]] = None
    ) -> str:
        if node_id in self.experiment.graph_context.node_ids:
            reaction = self.experiment.graph_context.node_ids[node_id].reaction
            hover_info = getattr(reaction, "hoverInfo", None)
            if isinstance(hover_info, str) and hover_info.strip():
                return hover_info.strip()

        metadata = data.get("metadata") if isinstance(data, dict) else None
        if isinstance(metadata, dict):
            for key in ("reactionHoverInfo", "hoverInfo"):
                hover_info = metadata.get(key)
                if isinstance(hover_info, str) and hover_info.strip():
                    return hover_info.strip()
        return ""

    def _reaction_context_for_node(
        self, node_id: str, data: Optional[dict[str, Any]] = None
    ) -> str:
        if node_id not in self.experiment.graph_context.node_ids:
            hover_info = self._reaction_hover_info_for_node(node_id, data)
            return f"Reaction hover information:\n{hover_info}" if hover_info else ""
        node = self.experiment.graph_context.node_ids[node_id]
        child_nodes = [
            nid
            for nid, parent in self.experiment.graph_context.parents.items()
            if parent == node_id
        ]
        reactants = [self.experiment.graph_context.node_ids[nid] for nid in child_nodes]
        reactants_str = "\n".join(reactant.smiles for reactant in reactants)
        reaction_str = f"Product: {node.smiles}\nReactants:\n{reactants_str}"
        hover_info = self._reaction_hover_info_for_node(node_id, data)
        if hover_info:
            reaction_str += f"\n\nReaction hover information:\n{hover_info}"
        return reaction_str

    @handles("chat-agent")
    async def handle_chat_agent(self, data: dict[str, Any]) -> None:
        self.setup_run_settings(data)
        agent_key = str(data.get("agentKey") or "")
        if not agent_key:
            raise ValueError("agentKey is required")
        query = str(data.get("query") or "").strip()
        if not query:
            raise ValueError("query is required")

        kind, _, target = agent_key.partition(":")
        routed_data = {**data, "query": query}
        if kind == "molecule":
            metadata = (
                data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
            )
            routed_data["nodeId"] = data.get("nodeId") or target
            routed_data["smiles"] = (
                data.get("smiles") or metadata.get("smiles") or target
            )
            await self.handle_custom_query_molecule(routed_data)
            return
        if kind == "reaction":
            routed_data["nodeId"] = data.get("nodeId") or target
            await self.handle_custom_query_reaction(routed_data)
            return
        raise ValueError(f"Unsupported chat agent key: {agent_key}")

    @handles("get-agent")
    async def handle_get_agent(self, data: dict[str, Any]) -> None:
        request = AgentRequest.model_validate(data)
        metadata = (
            data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        )

        record = None
        if metadata.get("kind") == "route-planning":
            result = self.experiment.route_planning_result
            plan_id = metadata.get("planId")
            record = (
                self._route_planning_planner_record(
                    result,
                    plan_id if isinstance(plan_id, str) else None,
                )
                if result is not None
                else None
            )

        if record is None:
            records = self.agent_records()
            if metadata.get("kind") == "route-planning":
                record = records.get("route-planning:planner")
            if record is None:
                record = records.get(request.agentKey) or AgentRecord()

        await self.websocket.send_json(
            AgentResponse(agentKey=request.agentKey, agent=record).model_dump(
                exclude_none=True
            )
        )

    @handles("load-context")
    async def handle_load_state(self, data: dict, *args, **kwargs) -> None:
        experiment_context = data.get("experimentContext") or {}
        if data.get("routePlanningResult") is not None and not experiment_context.get(
            "routePlanningResult"
        ):
            data = {
                **data,
                "experimentContext": {
                    **experiment_context,
                    "routePlanningResult": data["routePlanningResult"],
                },
            }

        await super().handle_load_state(data, *args, **kwargs)

        # Handle legacy experiments. The previous iteration of saved
        # context from the front-end included "nodes" and "edges" in
        # the top-level `data` dictionary -- they are NOT moved under
        # "experimentContext". It would be good to preserve the
        # back-compatibility with old experiments, and it's not a
        # terrible amount of effort to do so. The two options are to
        # manipulate `data` to put things in the current place, or to
        # just call the legacy function directly. This opts for the
        # latter approach.
        problem_type = data.get("problemType")
        if (
            problem_type == "retrosynthesis"
            and self.experiment.graph_context.is_empty()
        ):
            self.experiment.graph_context.load_state(data)
