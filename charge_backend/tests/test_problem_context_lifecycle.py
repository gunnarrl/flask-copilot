import argparse
import asyncio
from types import SimpleNamespace

from charge.clients.agent import AgentRuntimeConfig
from charge.experiments.experiment import AgentRegistryEntry

from charge_backend.backend_helper_funcs import Node
from charge_backend.backend_manager import FlaskActionManager
from charge_backend.flask_experiment import FlaskExperiment, GraphContext
from charge_backend.retrosynthesis import route_planner


class FakeWebSocket:
    headers = {}

    async def send_json(self, payload):
        self.last_payload = payload


def make_manager():
    websocket = FakeWebSocket()
    return FlaskActionManager(
        websocket=websocket,
        args=argparse.Namespace(backend="openai", model="gpt-4o-mini"),
        username="test-user",
    )


def route_procedure_step() -> route_planner.RouteProcedureStep:
    return route_planner.RouteProcedureStep(
        step_label="Step 1",
        product_smiles="CC(=O)Oc1ccccc1C(=O)O",
        precursor_smiles=["O=C(O)c1ccccc1O"],
        description="Acetylate salicylic acid.",
        transformation_type="acetylation",
        evidence_level="template",
        evidence_basis="Template route evidence.",
        source_refs=["Route 1"],
        conditions_summary="Template-only; conditions not provided.",
        yield_summary="No yield reported.",
        confidence="high",
        main_uncertainty="Forward conditions need verification.",
    )


def make_route_planning_result_payload() -> dict:
    result = route_planner.RoutePlanningResult(
        target_smiles="CC(=O)Oc1ccccc1C(=O)O",
        candidate_routes=[
            route_planner.CandidateRoutePlan(
                plan_id="plan_1",
                plan=route_planner.RoutePlanContent(
                    title="Aspirin from salicylic acid",
                    route_type="template_based",
                    route_strategy="Acetylate salicylic acid.",
                    route_value="Tests route-planning context lifecycle.",
                    evidence_overview="Template route evidence.",
                    novelty_rationale="Fixture route; no novelty.",
                    procedure_steps=[route_procedure_step()],
                ),
            )
        ],
    )
    return result.model_dump(mode="json")


def test_reset_problem_context_clears_retrosynthesis_state_for_lmo():
    manager = make_manager()
    manager.experiment.agent_registry["reaction:node_0"] = AgentRegistryEntry(
        agent=SimpleNamespace(task=None, save_memory=lambda: "old"),
        runtime_config=AgentRuntimeConfig(),
    )

    manager.reset_problem_context("optimization")

    assert manager.experiment.graph_context.is_empty()
    assert "agentSessions" not in manager.experiment.save_state()


def test_load_retrosynthesis_state_replaces_existing_graph():
    async def run() -> None:
        manager = make_manager()
        await manager.experiment.graph_context.add_node(
            Node(
                id="old_node",
                smiles="CCO",
                label="old",
                hoverInfo="old",
                level=0,
            )
        )

        await manager.handle_load_state(
            {
                "problemType": "retrosynthesis",
                "nodes": [
                    Node(
                        id="new_node",
                        smiles="CCN",
                        label="new",
                        hoverInfo="new",
                        level=0,
                    ).json()
                ],
                "edges": [],
            }
        )

        assert list(manager.experiment.graph_context.node_ids) == ["new_node"]

    asyncio.run(run())


def test_route_planning_result_round_trips_through_load_context():
    async def run() -> None:
        manager = make_manager()
        saved = FlaskExperiment(task=None)
        saved.route_planning_result = route_planner.RoutePlanningResult.model_validate(
            make_route_planning_result_payload()
        )
        saved.route_planning_result.base_branch_state = {"graphContext": {}}

        await manager.handle_load_state(
            {
                "problemType": "route-planning",
                "experimentContext": saved.save_state(),
            }
        )

        assert manager.experiment.route_planning_result is not None
        assert (
            manager.experiment.route_planning_result.candidate_routes[0].plan_id
            == "plan_1"
        )
        assert manager.experiment.route_planning_result.base_branch_state == {
            "graphContext": {}
        }

    asyncio.run(run())
