"""Draft route-level retrosynthesis plans for user review."""

import asyncio
import json
from typing import Awaitable, Callable, Literal, TYPE_CHECKING
from agent_framework import Message
from pydantic import BaseModel, Field
from lc_conductor import ToolRuntime
from flask_tools.chemistry.smiles_utils import canonicalize_smiles

from .context import RouteContext, RouteContextItem

if TYPE_CHECKING:
    from charge.clients.agent import AgentCallbackType
    from charge_backend.flask_experiment import FlaskExperiment


RoutePlanType = Literal["template_based", "hybrid", "new_proposal"]


class RouteProcedureStep(BaseModel):
    step_label: str
    product_smiles: str
    precursor_smiles: list[str]
    description: str
    transformation_type: str
    evidence_level: Literal["patent", "template", "tool", "proposal"]
    evidence_basis: str
    source_refs: list[str]
    conditions_summary: str
    yield_summary: str
    confidence: Literal["low", "medium", "high"]
    main_uncertainty: str


class RouteStep(BaseModel):
    step_id: str
    product_smiles: str
    precursor_smiles: list[str] = Field(min_length=1)
    canonical_product_smiles: str | None = None
    canonical_precursor_smiles: list[str | None] = Field(default_factory=list)
    reaction_smiles: str | None = None
    rationale: str | None = None


class RouteEvaluationFixOption(BaseModel):
    label: str
    description: str
    recommended: bool = False


class RouteEvaluationWarning(BaseModel):
    severity: Literal["low", "medium", "high"]
    step_id: str | None = None
    reason: str


class RouteEvaluationIssue(BaseModel):
    severity: Literal["medium", "high"]
    step_id: str | None = None
    reason: str
    fix_options: list[RouteEvaluationFixOption] = Field(min_length=1, max_length=3)


class RouteEvaluationOutputSchema(BaseModel):
    accepted: bool
    summary: str
    step_evidence: dict[str, str] = Field(default_factory=dict)
    issues: list[RouteEvaluationIssue] = Field(default_factory=list)
    warnings: list[RouteEvaluationWarning] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RoutePlanContent(BaseModel):
    title: str
    route_type: RoutePlanType
    route_strategy: str
    route_value: str
    evidence_overview: str
    novelty_rationale: str
    source_route_numbers: list[int] = Field(default_factory=list)
    procedure_steps: list[RouteProcedureStep] = Field(min_length=1)
    key_disconnections: list[str] = Field(default_factory=list)
    proposed_starting_materials: list[str] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)
    next_checks: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class CandidateRoutePlan(BaseModel):
    plan_id: str | None = None
    plan: RoutePlanContent
    route_steps: list[RouteStep] = Field(default_factory=list)
    tool_findings_summary: list[str] = Field(default_factory=list)
    evaluation: RouteEvaluationOutputSchema | None = None
    needs_user_decision: bool = False
    branch_state: dict | None = None
    answer: str | None = None


class RoutePlanDraft(BaseModel):
    plan: RoutePlanContent
    route_steps: list[RouteStep] = Field(min_length=1)
    tool_findings_summary: list[str] = Field(default_factory=list)


class RoutePlanningOutputSchema(BaseModel):
    reasoning_summary: str
    tool_findings_summary: list[str] = Field(default_factory=list)
    candidate_routes: list[RoutePlanDraft] = Field(min_length=1, max_length=5)


class RoutePlanningResult(BaseModel):
    target_smiles: str
    user_constraints: str | None = None
    route_context_items: list[RouteContextItem] = Field(default_factory=list)
    reasoning_summary: str = ""
    tool_findings_summary: list[str] = Field(default_factory=list)
    candidate_routes: list[CandidateRoutePlan] = Field(default_factory=list)
    base_branch_state: dict | None = None
    selected_plan_id: str | None = None


class RouteEvaluationDecision(BaseModel):
    result: RoutePlanningResult
    plan_id: str
    accepted: bool
    needs_user_decision: bool = False
    message: str | None = None


ROUTE_PLANNER_SYSTEM_PROMPT = (
    "You are a chemistry retrosynthesis planning assistant. Draft route-level "
    "options and concrete reaction-tree steps for a human chemist to review."
)

ROUTE_EVALUATOR_SYSTEM_PROMPT = (
    "You evaluate whether a proposed retrosynthesis route is coherent and "
    "chemically plausible enough for a human chemist to pursue or review. "
    "Do not require a complete laboratory procedure. Distinguish concrete "
    "route defects from non-blocking uncertainty, and recommend the smallest "
    "viable route correction when a defect exists."
)

ROUTE_PLAN_QA_SYSTEM_PROMPT = (
    "You answer questions about a proposed retrosynthesis route. Be concise, "
    "factual, and use only the provided plan and route evidence."
)


def build_route_planning_prompt(
    target_smiles: str,
    route_context: RouteContext,
    user_request: str | None = None,
) -> str:
    prompt = f"""Target:
- SMILES: `{target_smiles}`

Task:
Draft 5 detailed retrosynthetic route dossiers for user review. Use the
candidate template routes below as evidence, but do not assume they are
experimentally validated unless the route summary gives patent or tool evidence.

Requirements:
1. Propose route-level plans with enough detail for a chemist to judge the
   strategy, evidence, likely operations, and missing checks.
2. Write each plan as a route dossier, not a short outline. Preserve useful
   patent details from the route summaries: source references, reagents,
   solvents, additives, temperature, time, workup, purification, yield, scale,
   and operational cautions when present.
3. Each route must be one of: template_based, hybrid, new_proposal.
4. Use hybrid for routes that change one template route or combine useful ideas
   from multiple template routes.
5. Do not invent reaction conditions, yields, mechanisms, literature support,
   or purchasability claims that are not in the provided context.
6. For each route, explain the main rationale, key risks, and next checks.
7. For each route, list source_route_numbers using the route numbers from the
   evidence, such as [1] or [1, 3].
8. If proposing a new route, explain what template-route evidence inspired it.
9. Return route options for human approval, denial, or revision before any
   in-depth predictions are run.
10. Consecutive applications of the same template have already been contracted
   into one route step in the evidence. Keep those contractions as one
   reaction in your plans.
11. For each plan, output route_steps with exact product_smiles and
    precursor_smiles so the plan forms a reaction tree. Use reaction direction
    precursor1.precursor2 -> product.
12. When evidence allows, include at least two hybrid or new_proposal routes
    that are not direct copies of a single template route.
13. Prefer routes with practical, commonly available starting materials. The
    buyable-leaf ratio is only a coarse signal; use chemistry judgment and
    available tools when deciding whether leaves make good starting materials.
14. Explore useful alternatives before settling on the 5 plans. If a promising
    precursor or intermediate needs more template evidence, consider using the
    enumerate_template_routes tool on it.
15. Use verify_smiles and canonicalize_smiles when you need to check or clean
    product and precursor SMILES.
16. Use available MCP tools when they can materially improve the plan:
    reaction-search tools for uncertain disconnections, forward/retro prediction tools
    for hybrid or new-proposal steps, and synthesizability tools for questionable
    intermediates or starting materials.
17. Evidence priority is patents > route summaries/templates > tools/predictions
    > proposal assumptions. Clearly label any proposal assumptions; do not state
    them as facts.
18. Record durable facts learned from tool use in tool_findings_summary. Include
    important positive, negative, or inconclusive checks that should remain useful
    for later questions or refinements. Do not copy raw tool output, or include unimportant information.

Plan mix:
- Plan 1 should be the strongest evidence-backed route.
- Plan 2 should be the shortest/practical route if advanced intermediates are
  available.
- Plans 3 and 4 should be meaningfully different hybrids that change the main
  disconnection, order of operations, or sourcing strategy.
- Plan 5 should be an exploratory new_proposal when chemically reasonable; if a
  true new proposal is unreasonable, make it a clearly distinct hybrid and
  explain why.

For each procedure step, provide product_smiles, precursor_smiles,
transformation_type, evidence_level, evidence_basis, source_refs,
conditions_summary, yield_summary, confidence, and main_uncertainty. Use concise
plain text such as "No yield reported" or "Template-only; conditions not
provided" when a field is not supported by the evidence.

Ranking criteria:
- chemical plausibility
- practical, commonly available starting materials
- fewer questionable disconnections
- shorter and clearer routes
- useful diversity between route options

Candidate template routes:
{route_context.format()}
"""
    if user_request:
        prompt += f"\nAdditional user request:\n{user_request}\n"
    return prompt


async def plan_candidate_routes(
    target_smiles: str,
    route_context: RouteContext,
    experiment: "FlaskExperiment",
    tool_runtime: "ToolRuntime | None" = None,
    user_request: str | None = None,
    attachments: list[dict[str, object]] | None = None,
    agent_key: str | None = None,
    callback: "AgentCallbackType" = None,
) -> RoutePlanningOutputSchema:
    from charge.tasks.task import Task

    task_kwargs = tool_runtime.task_kwargs() if tool_runtime is not None else {}
    task = Task(
        system_prompt=ROUTE_PLANNER_SYSTEM_PROMPT,
        user_prompt=build_route_planning_prompt(
            target_smiles,
            route_context,
            user_request,
        ),
        structured_output_schema=RoutePlanningOutputSchema,
        attachments=attachments or [],
        **task_kwargs,
    )
    agent = _create_route_agent(experiment, task, agent_key, callback)
    output = await agent.run()
    return RoutePlanningOutputSchema.model_validate_json(output)


def _create_route_agent(
    experiment: "FlaskExperiment",
    task,
    agent_key: str | None,
    callback: "AgentCallbackType" = None,
):
    if agent_key is None and callback is None:
        return experiment.create_agent_with_experiment_state(task=task)
    kwargs = {}
    if agent_key is not None:
        kwargs["agent_key"] = agent_key
    if callback is not None:
        kwargs["callback"] = callback
    return experiment.create_agent_with_experiment_state(task=task, **kwargs)


def build_route_evaluation_prompt(
    target_smiles: str,
    plan: CandidateRoutePlan,
    pipette_context: str | None = None,
    route_context_items: list[RouteContextItem] | None = None,
) -> str:
    plan_markdown = format_candidate_route_plan(plan)
    route_steps = format_route_steps_for_prompt(plan.route_steps)
    relevant_route_context = route_context_for_plan(plan, route_context_items or [])
    pipette_context = pipette_context or "Pipette checks were not run."
    return f"""Evaluate this materialized retrosynthesis plan.

Target SMILES:
{target_smiles}

Written plan:
{plan_markdown}

Materialized reaction tree:
{route_steps}

Pipette reaction checks:
{pipette_context}

Original route evidence:
{relevant_route_context}

Evaluation rules:
1. Accept the route when it is coherent and chemically plausible enough to
   pursue, review, or materialize; it need not be a complete laboratory procedure.
   Missing conditions, uncertain yield, limited precedent, low confidence, optional
   verification, and other non-invalidating concerns are warnings or notes and never
   prevent acceptance. Missing experimental detail alone is not an issue.
2. An issue is a concrete defect that requires changing the plan before pursuing
   it, such as invalid chemistry or connectivity, an implausible key transformation,
   an unusable starting material, or another major blocker. Identify the affected
   step when possible and explain the concrete failure. Reserve high severity for
   route-invalidating failures; medium severity requires a meaningful correction.
3. A clearly labeled template-only, hybrid, or proposed step is not automatically
   an issue when chemically plausible. Use available tools when they can materially
   check the route, but failed, unavailable, or inconclusive tools do not prove a
   defect.
4. Ignore extra balancing byproducts added by Pipette unless they change the
   intended transformation or target.
5. Give each issue 1-3 self-contained fix_options that directly describe a concrete
   revision the planner can apply. Prefer the smallest viable change, such as
   correcting one precursor, replacing one reaction, or adjusting one step, while
   preserving the target, viable strategy, evidence, and unaffected steps.
6. Do not suggest generic investigation, more tool use, or literature consultation
   as fixes; put those in warnings or notes. Mark exactly one option per issue as
   recommended=true: the least disruptive viable correction.
7. Return only accepted, summary, step_evidence, issues, warnings, and notes.
"""


def build_route_revision_prompt(
    plan: CandidateRoutePlan,
    evaluation: RouteEvaluationOutputSchema,
    user_guidance: str | None = None,
    selected_fix_options: list[RouteEvaluationFixOption] | None = None,
) -> str:
    issues = format_route_evaluation_issues_for_prompt(evaluation)
    selected_fixes = format_selected_fix_options_for_prompt(selected_fix_options or [])
    guidance = ""
    if user_guidance:
        guidance = f"""

User guidance:
{user_guidance}
"""
    return f"""Revise only this retrosynthesis route plan using the evaluator feedback.

Current plan:
{format_candidate_route_plan(plan)}

Evaluator summary:
{evaluation.summary}

Evaluator issues and fix options:
{issues}

Selected fix options:
{selected_fixes}
{guidance}

Revision rules:
1. Return one complete revised route plan.
2. Keep the same overall target and user intent from the previous planning context.
3. Apply only the selected fix options and explicit user guidance. Do not apply
   unselected fix options or make unrelated improvements.
4. Make the smallest viable change needed to apply each selected fix.
5. Preserve all unaffected steps, route strategy, evidence, and plan content.
   Update dependent fields only when the selected change makes that necessary.
6. Do not revise unrelated route options from the previous planning response.
7. Preserve contracted reactions as single plan steps.
8. Record any durable tool-derived findings in tool_findings_summary; do not copy
   raw tool output, or include non-important information.
"""


def build_route_plan_question_prompt(
    plan_id: str,
    question: str,
) -> str:
    return f"""Answer the user's question about {plan_id}.

User question:
{question}
"""


def build_route_refinement_prompt(
    plan_id: str,
    user_guidance: str,
) -> str:
    return f"""Revise only {plan_id} using the user's guidance.

User guidance:
{user_guidance}

Revision rules:
1. Return one complete revised route plan.
2. Preserve useful source_route_numbers when the evidence still applies.
3. Return concrete route_steps with product_smiles and precursor_smiles.
"""


def build_more_route_plans_prompt(
    user_feedback: str | None = None,
) -> str:
    prompt = """Generate exactly 5 additional retrosynthetic route plans.
Do not repeat the existing plans, route steps, or main disconnection patterns.
"""
    if user_feedback:
        prompt += f"\nUser feedback:\n{user_feedback}\n"
    return prompt


def format_route_evaluation_issues_for_prompt(
    evaluation: RouteEvaluationOutputSchema,
) -> str:
    if not evaluation.issues:
        return "- No specific evaluator issues were reported."
    lines = []
    for issue in evaluation.issues:
        label = issue.step_id or "route"
        detail = f"- {label} ({issue.severity}): {issue.reason}"
        lines.append(detail)
    return "\n".join(lines)


def format_selected_fix_options_for_prompt(
    selected_fix_options: list[RouteEvaluationFixOption],
) -> str:
    if not selected_fix_options:
        return "- No evaluator fix option was selected."
    return "\n".join(
        f"- {option.label}: {option.description}" for option in selected_fix_options
    )


def format_route_steps_for_prompt(route_steps: list[RouteStep]) -> str:
    if not route_steps:
        return "- No materialized route steps were provided."
    lines = []
    for step in route_steps:
        precursors = " + ".join(step.precursor_smiles)
        detail = f"- {step.step_id}: {precursors} -> {step.product_smiles}"
        if step.reaction_smiles:
            detail += f"; reaction_smiles={step.reaction_smiles}"
        if step.rationale:
            detail += f"; rationale={step.rationale}"
        lines.append(detail)
    return "\n".join(lines)


def route_context_for_plan(
    plan: CandidateRoutePlan,
    route_context_items: list[RouteContextItem],
) -> str:
    return RouteContext(items=route_context_items).format_selected(
        plan.plan.source_route_numbers
    )


async def build_pipette_reaction_context(
    plan: CandidateRoutePlan,
    callback: "AgentCallbackType" = None,
    status_callback: Callable[[str], Awaitable[None]] | None = None,
) -> str:
    reaction_steps = [
        (step.step_id, step.reaction_smiles)
        for step in plan.route_steps
        if step.reaction_smiles
    ]
    if not reaction_steps:
        return "No reaction SMILES were available for Pipette checks."
    total = len(reaction_steps)

    try:
        from flask_tools.pipette.grade_rxn import grade_reaction
    except Exception as exc:
        return f"Pipette unavailable: {type(exc).__name__}: {exc}"

    async def grade_one(index: int, step_id: str, reaction: str):
        call_id = f"pipette:{plan.plan_id}:{step_id}"
        if status_callback is not None:
            await status_callback(
                f"Pipette started checking {step_id} ({index} of {total})."
            )
        if callback is not None:
            await callback.on_tool_call(
                "grade_reaction",
                {"rxn_smiles": reaction},
                source="Pipette",
                call_id=call_id,
            )
        try:
            grade = (await asyncio.to_thread(grade_reaction, reaction))[0]
        except Exception as exc:
            if callback is not None:
                await callback.on_tool_result(
                    "grade_reaction",
                    f"{type(exc).__name__}: {exc}",
                    is_error=True,
                    source="Pipette",
                    call_id=call_id,
                )
            if status_callback is not None:
                await status_callback(
                    f"Pipette failed {step_id} ({index} of {total})."
                )
            return exc
        if callback is not None:
            await callback.on_tool_result(
                "grade_reaction",
                grade.model_dump(mode="json"),
                source="Pipette",
                call_id=call_id,
            )
        return grade

    results = await asyncio.gather(
        *(
            grade_one(index, step_id, reaction)
            for index, (step_id, reaction) in enumerate(reaction_steps, start=1)
        )
    )

    lines = [
        "Pipette fixed/balanced reaction SMILES are advisory only; ignore extra "
        "balancing byproducts when judging the planned route tree."
    ]
    for (step_id, reaction), result in zip(reaction_steps, results, strict=True):
        if isinstance(result, BaseException):
            lines.append(
                f"- {step_id}: Pipette failed: {type(result).__name__}: {result}; "
                f"original={reaction}"
            )
            continue
        grade = result
        lines.append(
            f"- {step_id}: {grade.final_grade}; {grade.short_comment}; original={reaction}"
        )
        fixed_reaction = pipette_fixed_reaction_smiles(grade)
        if fixed_reaction and fixed_reaction != reaction:
            lines.append(f"  fixed_or_balanced={fixed_reaction}")
        if grade.comment:
            lines.append(f"  comment={grade.comment}")
        for result in grade.results:
            status = getattr(result.status, "value", str(result.status))
            if status in {"fail", "error", "unknown"} or result.comment:
                lines.append(f"  {result.name}: {status} - {result.comment}")
    return "\n".join(lines)


def pipette_fixed_reaction_smiles(grade) -> str | None:
    for result in getattr(grade, "results", []):
        if result.name != "llm_reaction_fix" or result.data is None:
            continue
        fixed = getattr(result.data, "fixed_reaction_smiles", None)
        if isinstance(fixed, str) and fixed:
            return fixed
    return None


async def evaluate_route_plan(
    target_smiles: str,
    plan: CandidateRoutePlan,
    experiment: "FlaskExperiment",
    tool_runtime: "ToolRuntime | None" = None,
    route_context_items: list[RouteContextItem] | None = None,
    agent_key: str | None = "route-planning:evaluator",
    callback: "AgentCallbackType" = None,
    pipette_status_callback: Callable[[str], Awaitable[None]] | None = None,
    status_callback: Callable[[str], Awaitable[None]] | None = None,
) -> RouteEvaluationOutputSchema:
    from charge.tasks.task import Task

    task_kwargs = tool_runtime.task_kwargs() if tool_runtime is not None else {}
    pipette_context = await build_pipette_reaction_context(
        plan,
        callback,
        pipette_status_callback,
    )
    task = Task(
        system_prompt=ROUTE_EVALUATOR_SYSTEM_PROMPT,
        user_prompt=build_route_evaluation_prompt(
            target_smiles,
            plan,
            pipette_context,
            route_context_items,
        ),
        structured_output_schema=RouteEvaluationOutputSchema,
        **task_kwargs,
    )
    agent = _create_route_agent(experiment, task, agent_key, callback)
    progress_task = None
    if status_callback is not None:
        await status_callback(f"Evaluator started reviewing {plan.plan_id}.")

        async def report_progress() -> None:
            elapsed = 0
            while True:
                await asyncio.sleep(30)
                elapsed += 30
                await status_callback(
                    f"Evaluator is still reviewing {plan.plan_id} "
                    f"({elapsed} seconds elapsed)."
                )

        progress_task = asyncio.create_task(report_progress())
    try:
        output = await agent.run()
        evaluation = RouteEvaluationOutputSchema.model_validate_json(output)
    finally:
        if progress_task is not None:
            progress_task.cancel()
            await asyncio.gather(progress_task, return_exceptions=True)
    plan.evaluation = evaluation
    plan.needs_user_decision = False
    if status_callback is not None:
        await status_callback(f"Evaluator completed review of {plan.plan_id}.")
    return evaluation


async def revise_route_plan_from_evaluation(
    plan: CandidateRoutePlan,
    evaluation: RouteEvaluationOutputSchema,
    experiment: "FlaskExperiment",
    tool_runtime: "ToolRuntime | None" = None,
    agent_key: str | None = None,
    user_guidance: str | None = None,
    selected_fix_options: list[RouteEvaluationFixOption] | None = None,
    callback: "AgentCallbackType" = None,
) -> CandidateRoutePlan:
    from charge.tasks.task import Task

    task_kwargs = tool_runtime.task_kwargs() if tool_runtime is not None else {}
    task = Task(
        system_prompt=ROUTE_PLANNER_SYSTEM_PROMPT,
        user_prompt=build_route_revision_prompt(
            plan,
            evaluation,
            user_guidance,
            selected_fix_options,
        ),
        structured_output_schema=RoutePlanDraft,
        **task_kwargs,
    )
    agent = _create_route_agent(experiment, task, agent_key, callback)
    output = await agent.run()
    revised = RoutePlanDraft.model_validate_json(output)
    return CandidateRoutePlan(
        plan_id=plan.plan_id,
        plan=revised.plan,
        route_steps=canonicalize_route_steps(revised.route_steps),
        tool_findings_summary=revised.tool_findings_summary,
    )


async def answer_route_plan_question(
    result: RoutePlanningResult,
    plan_id: str,
    question: str,
    experiment: "FlaskExperiment",
    tool_runtime: "ToolRuntime | None" = None,
    agent_key: str | None = "route-planning:planner",
    callback: "AgentCallbackType" = None,
) -> str:
    from charge.tasks.task import Task

    _, plan = find_route_plan(result, plan_id)
    task_kwargs = tool_runtime.task_kwargs() if tool_runtime is not None else {}
    task = Task(
        system_prompt=ROUTE_PLAN_QA_SYSTEM_PROMPT,
        user_prompt=build_route_plan_question_prompt(plan_id, question),
        **task_kwargs,
    )
    agent = _create_route_agent(experiment, task, agent_key, callback)
    answer = str(await agent.run()).strip()
    plan.answer = answer
    return answer


async def refine_route_plan_from_user_guidance(
    result: RoutePlanningResult,
    plan_id: str,
    user_guidance: str,
    experiment: "FlaskExperiment",
    tool_runtime: "ToolRuntime | None" = None,
    agent_key: str | None = "route-planning:planner",
    callback: "AgentCallbackType" = None,
) -> CandidateRoutePlan:
    from charge.tasks.task import Task

    plan_index, _ = find_route_plan(result, plan_id)
    task_kwargs = tool_runtime.task_kwargs() if tool_runtime is not None else {}
    task = Task(
        system_prompt=ROUTE_PLANNER_SYSTEM_PROMPT,
        user_prompt=build_route_refinement_prompt(plan_id, user_guidance),
        structured_output_schema=RoutePlanDraft,
        **task_kwargs,
    )
    agent = _create_route_agent(experiment, task, agent_key, callback)
    revised = RoutePlanDraft.model_validate_json(await agent.run())
    result.candidate_routes[plan_index] = CandidateRoutePlan(
        plan_id=plan_id,
        plan=revised.plan,
        route_steps=canonicalize_route_steps(revised.route_steps),
        tool_findings_summary=revised.tool_findings_summary,
    )
    return result.candidate_routes[plan_index]


async def plan_more_candidate_routes(
    result: RoutePlanningResult,
    experiment: "FlaskExperiment",
    tool_runtime: "ToolRuntime | None" = None,
    user_feedback: str | None = None,
    agent_key: str | None = "route-planning:planner",
    callback: "AgentCallbackType" = None,
) -> RoutePlanningOutputSchema:
    from charge.tasks.task import Task

    task_kwargs = tool_runtime.task_kwargs() if tool_runtime is not None else {}
    task = Task(
        system_prompt=ROUTE_PLANNER_SYSTEM_PROMPT,
        user_prompt=build_more_route_plans_prompt(user_feedback),
        structured_output_schema=RoutePlanningOutputSchema,
        **task_kwargs,
    )
    agent = _create_route_agent(experiment, task, agent_key, callback)
    output = RoutePlanningOutputSchema.model_validate_json(await agent.run())
    append_route_plans(result, output)
    return output


async def evaluate_selected_route_plan(
    result: RoutePlanningResult,
    plan_id: str,
    experiment: "FlaskExperiment",
    tool_runtime: "ToolRuntime | None" = None,
    evaluator_agent_key: str | None = "route-planning:evaluator",
    evaluator_callback: "AgentCallbackType" = None,
    pipette_status_callback: Callable[[str], Awaitable[None]] | None = None,
    status_callback: Callable[[str], Awaitable[None]] | None = None,
) -> RouteEvaluationDecision:
    _, plan = find_route_plan(result, plan_id)
    evaluation = await evaluate_route_plan(
        result.target_smiles,
        plan,
        experiment,
        tool_runtime,
        route_context_items=result.route_context_items,
        agent_key=evaluator_agent_key,
        callback=evaluator_callback,
        pipette_status_callback=pipette_status_callback,
        status_callback=status_callback,
    )
    return route_evaluation_decision(result, plan_id, plan, evaluation)


async def apply_evaluator_fixes_to_route_plan(
    result: RoutePlanningResult,
    plan_id: str,
    experiment: "FlaskExperiment",
    tool_runtime: "ToolRuntime | None" = None,
    planner_agent_key: str | None = "route-planning:planner",
    evaluator_agent_key: str | None = "route-planning:evaluator",
    user_guidance: str | None = None,
    selected_fix_options: list[RouteEvaluationFixOption] | None = None,
    planner_callback: "AgentCallbackType" = None,
    evaluator_callback: "AgentCallbackType" = None,
    pipette_status_callback: Callable[[str], Awaitable[None]] | None = None,
    status_callback: Callable[[str], Awaitable[None]] | None = None,
) -> RouteEvaluationDecision:
    plan_index, plan = find_route_plan(result, plan_id)
    if plan.evaluation is None:
        raise ValueError(f"Route plan has no evaluator feedback: {plan_id}")

    revised_plan = await revise_route_plan_from_evaluation(
        plan,
        plan.evaluation,
        experiment,
        tool_runtime,
        agent_key=planner_agent_key,
        user_guidance=user_guidance,
        selected_fix_options=selected_fix_options,
        callback=planner_callback,
    )
    result.candidate_routes[plan_index] = revised_plan

    evaluation = await evaluate_route_plan(
        result.target_smiles,
        revised_plan,
        experiment,
        tool_runtime,
        route_context_items=result.route_context_items,
        agent_key=evaluator_agent_key,
        callback=evaluator_callback,
        pipette_status_callback=pipette_status_callback,
        status_callback=status_callback,
    )
    return route_evaluation_decision(result, plan_id, revised_plan, evaluation)


def continue_with_evaluated_route_plan(
    result: RoutePlanningResult,
    plan_id: str,
) -> RouteEvaluationDecision:
    _, plan = find_route_plan(result, plan_id)
    plan.needs_user_decision = False
    accepted = plan.evaluation.accepted if plan.evaluation else False
    return RouteEvaluationDecision(
        result=result,
        plan_id=plan_id,
        accepted=accepted,
        needs_user_decision=False,
        message=None,
    )


def find_route_plan(
    result: RoutePlanningResult,
    plan_id: str,
) -> tuple[int, CandidateRoutePlan]:
    for index, plan in enumerate(result.candidate_routes):
        if plan.plan_id == plan_id:
            return index, plan
    raise ValueError(f"Route plan not found: {plan_id}")


def route_evaluation_decision(
    result: RoutePlanningResult,
    plan_id: str,
    plan: CandidateRoutePlan,
    evaluation: RouteEvaluationOutputSchema,
) -> RouteEvaluationDecision:
    needs_decision = bool(evaluation.issues)
    if needs_decision:
        evaluation.accepted = False
    plan.needs_user_decision = needs_decision
    return RouteEvaluationDecision(
        result=result,
        plan_id=plan_id,
        accepted=evaluation.accepted,
        needs_user_decision=needs_decision,
        message=route_evaluation_decision_message(evaluation) if needs_decision else None,
    )


def route_evaluation_decision_message(evaluation: RouteEvaluationOutputSchema) -> str:
    reasons = [issue.reason for issue in evaluation.issues]
    reason_text = "; ".join(reasons) if reasons else evaluation.summary
    status = (
        "accepted the plan but found issues"
        if evaluation.accepted
        else "did not accept the plan"
    )
    return (
        f"Evaluator {status} for the following reasons: "
        f"{reason_text}. Would you like to continue or apply fixes?"
    )


def canonicalize_route_steps(route_steps: list[RouteStep]) -> list[RouteStep]:
    canonicalized_steps = []
    seen_step_ids = set()
    for index, step in enumerate(route_steps, start=1):
        step_id = step.step_id or f"step_{index}"
        if step_id in seen_step_ids:
            step_id = f"{step_id}_{index}"
        seen_step_ids.add(step_id)

        canonical_product = _canonical_smiles_or_none(step.product_smiles)
        canonical_precursors = [
            _canonical_smiles_or_none(precursor) for precursor in step.precursor_smiles
        ]
        reaction_product = canonical_product or step.product_smiles
        reaction_precursors = [
            canonical or precursor
            for canonical, precursor in zip(canonical_precursors, step.precursor_smiles)
        ]

        canonicalized_steps.append(
            RouteStep(
                step_id=step_id,
                product_smiles=step.product_smiles,
                precursor_smiles=step.precursor_smiles,
                canonical_product_smiles=canonical_product,
                canonical_precursor_smiles=canonical_precursors,
                reaction_smiles=f"{'.'.join(reaction_precursors)}>>{reaction_product}",
                rationale=step.rationale,
            )
        )
    return canonicalized_steps


def _canonical_smiles_or_none(smiles: str) -> str | None:
    try:
        canonical = canonicalize_smiles(smiles)
    except Exception:
        return None
    if canonical == "Invalid SMILES":
        return None
    return canonical


def append_route_plans(
    result: RoutePlanningResult,
    output: RoutePlanningOutputSchema,
) -> list[CandidateRoutePlan]:
    plan_numbers = []
    for plan in result.candidate_routes:
        if not plan.plan_id or not plan.plan_id.startswith("plan_"):
            continue
        plan_numbers.append(int(plan.plan_id.removeprefix("plan_")))
    start_index = max(plan_numbers, default=len(result.candidate_routes)) + 1

    new_plans = [
        CandidateRoutePlan(
            plan_id=f"plan_{start_index + offset}",
            plan=route.plan,
            route_steps=canonicalize_route_steps(route.route_steps),
            tool_findings_summary=route.tool_findings_summary,
        )
        for offset, route in enumerate(output.candidate_routes)
    ]
    result.candidate_routes.extend(new_plans)
    result.tool_findings_summary.extend(output.tool_findings_summary)
    if output.reasoning_summary:
        result.reasoning_summary = "\n\n".join(
            text
            for text in [result.reasoning_summary, output.reasoning_summary]
            if text
        )
    return new_plans


def format_candidate_route_plan(
    route: CandidateRoutePlan | RoutePlanContent, index: int | None = None
) -> str:
    plan = route.plan if isinstance(route, CandidateRoutePlan) else route
    lines = []
    heading = plan.title if index is None else f"{index}. {plan.title}"
    lines.extend(
        [
            f"## {heading}",
            f"- Type: {plan.route_type}",
        ]
    )
    lines.append(f"- Strategy: {plan.route_strategy}")
    if plan.key_disconnections:
        lines.append(f"- Key disconnections: {'; '.join(plan.key_disconnections)}")
    lines.append(f"- Route value: {plan.route_value}")
    lines.append(f"- Evidence overview: {plan.evidence_overview}")
    lines.append(f"- Novelty rationale: {plan.novelty_rationale}")
    if plan.source_route_numbers:
        routes = "; ".join(f"Route {number}" for number in plan.source_route_numbers)
        lines.append(f"- Source routes: {routes}")
    if plan.proposed_starting_materials:
        lines.append(
            f"- Proposed starting materials: {'; '.join(plan.proposed_starting_materials)}"
        )
    if plan.key_risks:
        lines.append(f"- Key risks: {'; '.join(plan.key_risks)}")
    if plan.next_checks:
        lines.append(f"- Next checks: {'; '.join(plan.next_checks)}")
    if plan.assumptions:
        lines.append(f"- Assumptions: {'; '.join(plan.assumptions)}")
    lines.append("- Procedure outline:")
    for index, step in enumerate(plan.procedure_steps, start=1):
        lines.append(f"  {index}. {step.step_label} ({step.confidence} confidence)")
        lines.append(f"     - Product: {step.product_smiles}")
        lines.append(f"     - Precursors: {' + '.join(step.precursor_smiles)}")
        lines.append(f"     - Transformation: {step.transformation_type}")
        lines.append(f"     - Description: {step.description}")
        lines.append(f"     - Evidence: {step.evidence_level}; {step.evidence_basis}")
        if step.source_refs:
            lines.append(f"     - Sources: {'; '.join(step.source_refs)}")
        lines.append(f"     - Conditions: {step.conditions_summary}")
        lines.append(f"     - Yield: {step.yield_summary}")
        lines.append(f"     - Main uncertainty: {step.main_uncertainty}")
    return "\n".join(lines)


def route_planning_result_from_output(
    target_smiles: str,
    output: RoutePlanningOutputSchema,
    *,
    user_constraints: str | None = None,
    route_context: RouteContext | None = None,
) -> RoutePlanningResult:
    candidate_routes = [
        CandidateRoutePlan(
            plan_id=f"plan_{index}",
            plan=route.plan,
            route_steps=canonicalize_route_steps(route.route_steps),
            tool_findings_summary=route.tool_findings_summary,
        )
        for index, route in enumerate(output.candidate_routes, start=1)
    ]
    return RoutePlanningResult(
        target_smiles=target_smiles,
        user_constraints=user_constraints,
        route_context_items=route_context.items if route_context else [],
        reasoning_summary=output.reasoning_summary,
        tool_findings_summary=output.tool_findings_summary,
        candidate_routes=candidate_routes,
    )


def route_planning_result_payload(result: RoutePlanningResult) -> dict:
    return result.model_dump(
        mode="json",
        exclude={
            "base_branch_state": True,
            "candidate_routes": {"__all__": {"branch_state": True}},
        },
    )


def route_evaluation_decision_payload(decision: RouteEvaluationDecision) -> dict:
    payload = decision.model_dump(mode="json", exclude={"result": True})
    payload["result"] = route_planning_result_payload(decision.result)
    return payload


def save_route_planning_branch_state(experiment: "FlaskExperiment") -> dict:
    return experiment.save_state(include_route_planning_result=False)


def build_compact_route_planner_messages(
    result: RoutePlanningResult,
    *,
    latest_user_message: str | None = None,
    latest_assistant_message: str | None = None,
) -> list[dict]:
    route_context = RouteContext(items=result.route_context_items)
    messages = [
        Message(
            role="user",
            contents=[
                build_route_planning_prompt(
                result.target_smiles,
                route_context,
                result.user_constraints,
                )
            ],
        ).to_dict(),
        Message(
            role="assistant",
            contents=[build_compact_route_planner_response(result)],
        ).to_dict(),
    ]
    if latest_user_message:
        messages.append(
            Message(role="user", contents=[latest_user_message]).to_dict()
        )
    if latest_assistant_message:
        messages.append(
            Message(role="assistant", contents=[latest_assistant_message]).to_dict()
        )
    return messages


def build_compact_route_planner_response(
    result: RoutePlanningResult,
) -> str:
    lines = ["# Route Planner Compact State"]
    if result.reasoning_summary:
        lines.extend(["", "## Reasoning Summary", result.reasoning_summary])
    findings = [
        *result.tool_findings_summary,
        *[
            finding
            for route in result.candidate_routes
            for finding in route.tool_findings_summary
        ],
    ]
    if findings:
        lines.extend(["", "## Durable Tool Findings"])
        for finding in dict.fromkeys(findings):
            lines.append(f"- {finding}")
    lines.extend(["", "## Current Route Plans"])
    for index, route in enumerate(result.candidate_routes, start=1):
        lines.extend(
            [
                "",
                format_candidate_route_plan(route, index),
                "",
                "Materialized route steps:",
                format_route_steps_for_prompt(route.route_steps),
            ]
        )
    return "\n".join(lines)


def save_compact_route_planning_branch_state(
    experiment: "FlaskExperiment",
    result: RoutePlanningResult,
    *,
    latest_user_message: str | None = None,
    latest_assistant_message: str | None = None,
) -> dict:
    planner = experiment.agent_registry["route-planning:planner"].agent
    memory = json.loads(planner.save_memory())
    memory["state"]["in_memory"]["messages"] = build_compact_route_planner_messages(
        result,
        latest_user_message=latest_user_message,
        latest_assistant_message=latest_assistant_message,
    )
    planner.load_memory(json.dumps(memory))
    return save_route_planning_branch_state(experiment)


def restore_route_planning_branch_state(
    experiment: "FlaskExperiment",
    branch_state: dict | None,
    result: RoutePlanningResult,
) -> None:
    if branch_state is not None:
        experiment.load_state(branch_state)
    experiment.route_planning_result = result


def commit_route_plan_to_graph(
    result: RoutePlanningResult,
    plan_id: str,
    experiment: "FlaskExperiment",
    molecule_name_format: str = "brand",
):
    from charge_backend.backend_helper_funcs import (
        Edge,
        Node,
        Reaction,
        calculate_positions,
    )
    from charge_backend.flask_experiment import GraphContext
    from charge_backend.moleculedb.molecule_naming import smiles_to_html
    from charge_backend.moleculedb.purchasable import is_purchasable
    from charge_backend.retrosynthesis.mapping import build_mapped_reaction_dict_or_none

    _, plan = find_route_plan(result, plan_id)
    graph = GraphContext()
    node_ids_by_key = {}
    node_count = 0
    edge_count = 0

    def add_node(smiles: str, level: int, parent_id: str | None = None) -> str:
        nonlocal node_count, edge_count
        node_id = f"{plan_id}_node_{node_count}"
        node_count += 1
        sources = is_purchasable(smiles)
        purchasable = bool(sources)
        hover_info = (
            f"# Molecule\n\n**SMILES:** {smiles}\n\n"
            f"**Purchasable**? {'Yes' if purchasable else 'No'}"
        )
        if sources:
            hover_info += f" (via {', '.join(sources)})"
        node = Node(
            id=node_id,
            smiles=smiles,
            label=smiles_to_html(smiles, molecule_name_format),
            hoverInfo=hover_info,
            level=level,
            parentId=parent_id,
            purchasable=purchasable,
        )
        edge = None
        if parent_id is not None:
            edge = Edge(f"{plan_id}_edge_{edge_count}", parent_id, node_id, "complete")
            edge_count += 1
        graph._add_node(node, edge)
        return node_id

    def update_subtree_levels(node_id: str, level: int) -> None:
        graph.node_ids[node_id].level = level
        for child_id, parent_id in list(graph.parents.items()):
            if parent_id == node_id:
                update_subtree_levels(child_id, level + 1)

    def attach_existing_root(node_id: str, parent_id: str) -> bool:
        nonlocal edge_count
        node = graph.node_ids[node_id]
        if node.parentId == parent_id:
            return True
        if node.parentId is not None:
            return False
        node.parentId = parent_id
        graph.parents[node_id] = parent_id
        edge = Edge(f"{plan_id}_edge_{edge_count}", parent_id, node_id, "complete")
        edge_count += 1
        graph.edges[edge.id] = edge
        update_subtree_levels(node_id, graph.node_ids[parent_id].level + 1)
        graph.recalculate_nodes_per_level()
        return True

    for step in plan.route_steps:
        product_key = step.canonical_product_smiles or step.product_smiles
        product_id = node_ids_by_key.get(product_key)
        if product_id is None:
            product_id = add_node(step.product_smiles, 0)
            node_ids_by_key[product_key] = product_id

        product_node = graph.node_ids[product_id]
        product_node.reaction = Reaction(
            id=f"{plan_id}:{step.step_id}",
            hoverInfo=route_step_reaction_hover_info(plan.plan.title, step),
            highlight="red",
            label="FLASK",
            templatesSearched=False,
            mappedReaction=build_mapped_reaction_dict_or_none(
                reactants=step.precursor_smiles,
                products=[step.product_smiles],
                log_msg=(
                    "Failed to build route-planning mapped reaction for "
                    "plan_id={plan_id} step_id={step_id}"
                ),
                plan_id=plan_id,
                step_id=step.step_id,
            ),
        )
        for index, precursor in enumerate(step.precursor_smiles):
            precursor_key = (
                step.canonical_precursor_smiles[index]
                if index < len(step.canonical_precursor_smiles)
                else precursor
            ) or precursor
            precursor_id = node_ids_by_key.get(precursor_key)
            if precursor_id and attach_existing_root(precursor_id, product_id):
                continue
            precursor_id = add_node(precursor, product_node.level + 1, product_id)
            if precursor_key not in node_ids_by_key:
                node_ids_by_key[precursor_key] = precursor_id

    graph.recalculate_nodes_per_level()
    calculate_positions(list(graph.node_ids.values()))
    result.selected_plan_id = plan_id
    experiment.graph_context = graph
    experiment.route_planning_result = result
    return graph


def route_step_reaction_hover_info(plan_title: str, step: RouteStep) -> str:
    lines = [f"# {plan_title}", "", f"**Step:** {step.step_id}"]
    if step.reaction_smiles:
        lines.append(f"**Reaction SMILES:** {step.reaction_smiles}")
    if step.rationale:
        lines.extend(["", step.rationale])
    return "\n".join(lines)


def format_route_planning_output(output: RoutePlanningOutputSchema) -> str:
    lines = ["# Candidate Route Plans", "", output.reasoning_summary]

    for index, route in enumerate(output.candidate_routes, start=1):
        lines.extend(["", format_candidate_route_plan(route.plan, index)])

    return "\n".join(lines)
