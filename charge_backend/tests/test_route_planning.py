###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

import asyncio

from charge_backend.retrosynthesis import route_planner


def procedure_step(title="Assemble target"):
    return {
        "step_label": "Step 1",
        "product_smiles": "CCO",
        "precursor_smiles": ["CC=O"],
        "description": title,
        "transformation_type": "reduction",
        "evidence_level": "template",
        "evidence_basis": "Supported by route evidence.",
        "source_refs": ["Route 1"],
        "conditions_summary": "Conditions not provided.",
        "yield_summary": "No yield reported.",
        "confidence": "medium",
        "main_uncertainty": "Conditions need verification.",
    }


def route_draft(title="Template route"):
    return route_planner.RoutePlanDraft.model_validate(
        {
            "plan": {
                "title": title,
                "route_type": "template_based",
                "route_strategy": "Use the supported disconnection.",
                "route_value": "Direct route.",
                "evidence_overview": "Template evidence.",
                "novelty_rationale": "Established route.",
                "source_route_numbers": [1],
                "procedure_steps": [procedure_step()],
            },
            "route_steps": [
                {
                    "step_id": "step_1",
                    "product_smiles": "CCO",
                    "precursor_smiles": ["CC=O"],
                }
            ],
        }
    )


def planning_output(*titles):
    return route_planner.RoutePlanningOutputSchema(
        reasoning_summary="Compared the available routes.",
        candidate_routes=[route_draft(title) for title in titles],
    )


class FakeAgent:
    def __init__(self, output):
        self.output = output

    async def run(self):
        return self.output


class FakeExperiment:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.tasks = []
        self.agent_keys = []

    def create_agent_with_experiment_state(self, task, agent_key=None, **kwargs):
        self.tasks.append(task)
        self.agent_keys.append(agent_key)
        return FakeAgent(self.outputs.pop(0))


def make_result(*titles):
    return route_planner.route_planning_result_from_output("CCO", planning_output(*titles))


def test_planner_output_becomes_stable_materialized_candidates():
    experiment = FakeExperiment(planning_output("Route A", "Route B").model_dump_json())

    output = asyncio.run(
        route_planner.plan_candidate_routes(
            "CCO",
            route_planner.RouteContext(),
            experiment,
            agent_key="route-planning:planner",
        )
    )
    result = route_planner.route_planning_result_from_output("CCO", output)

    assert [plan.plan_id for plan in result.candidate_routes] == ["plan_1", "plan_2"]
    assert result.candidate_routes[0].route_steps[0].reaction_smiles == "CC=O>>CCO"
    assert experiment.agent_keys == ["route-planning:planner"]


def test_refinement_replaces_only_the_selected_plan():
    result = make_result("Original A", "Original B")
    result.selected_plan_id = "plan_2"
    result.candidate_routes[0].materialized = True
    experiment = FakeExperiment(route_draft("Revised A").model_dump_json())

    revised = asyncio.run(
        route_planner.refine_route_plan_from_user_guidance(
            result,
            "plan_1",
            "Use the supported precursor.",
            experiment,
        )
    )

    assert revised.plan.title == "Revised A"
    assert revised.materialized is False
    assert result.candidate_routes[1].plan.title == "Original B"
    assert result.selected_plan_id == "plan_2"
    assert experiment.agent_keys == ["route-planning:planner"]


def blocking_evaluation():
    return route_planner.RouteEvaluationOutputSchema.model_validate(
        {
            "accepted": False,
            "summary": "The disconnection is unsupported.",
            "issues": [
                {
                    "severity": "high",
                    "step_id": "step_1",
                    "reason": "The first disconnection is unsupported",
                    "fix_options": [
                        {
                            "label": "Use supported precursor",
                            "description": "Replace the first precursor.",
                            "recommended": True,
                        }
                    ],
                }
            ],
        }
    )


def test_blocking_evaluation_requires_a_user_decision():
    result = make_result("Original")
    experiment = FakeExperiment(blocking_evaluation().model_dump_json())
    statuses = []

    async def run():
        async def record_status(message):
            statuses.append(message)

        return await route_planner.evaluate_selected_route_plan(
            result,
            "plan_1",
            experiment,
            status_callback=record_status,
        )

    decision = asyncio.run(run())

    assert decision.accepted is False
    assert decision.needs_user_decision is True
    assert "first disconnection is unsupported" in decision.message
    assert result.candidate_routes[0].evaluation == blocking_evaluation()
    assert statuses == [
        "Evaluator started reviewing plan_1.",
        "Evaluator completed review of plan_1.",
    ]


def test_applying_evaluator_fix_returns_an_unevaluated_revised_plan():
    result = make_result("Original")
    result.candidate_routes[0].evaluation = blocking_evaluation()
    experiment = FakeExperiment(route_draft("Fixed route").model_dump_json())

    revised = asyncio.run(
        route_planner.apply_evaluator_fixes_to_route_plan(result, "plan_1", experiment)
    )

    assert revised is result.candidate_routes[0]
    assert revised.plan_id == "plan_1"
    assert revised.plan.title == "Fixed route"
    assert revised.evaluation is None
    assert revised.needs_user_decision is False
    assert experiment.agent_keys == ["route-planning:planner"]


def test_materializing_plans_builds_graph_and_preserves_materialization_history(
    monkeypatch,
):
    monkeypatch.setattr(
        "charge_backend.moleculedb.purchasable.is_purchasable", lambda smiles: []
    )
    monkeypatch.setattr(
        "charge_backend.retrosynthesis.mapping.build_mapped_reaction_dict_or_none",
        lambda **kwargs: {
            "reactants": kwargs["reactants"],
            "products": kwargs["products"],
        },
    )
    result = make_result("Two-step route", "Other route")
    result.candidate_routes[0].route_steps = [
        route_planner.RouteStep(
            step_id="step_1",
            product_smiles="CCO",
            precursor_smiles=["CC=O"],
        ),
        route_planner.RouteStep(
            step_id="step_2",
            product_smiles="CC=O",
            precursor_smiles=["CC"],
        ),
    ]

    experiment = type("Experiment", (), {})()
    graph = route_planner.commit_route_plan_to_graph(result, "plan_1", experiment)

    assert result.selected_plan_id == "plan_1"
    assert result.candidate_routes[0].materialized is True
    assert result.candidate_routes[1].materialized is False
    assert len(graph.node_ids) == 3
    assert len(graph.edges) == 2
    assert sorted(node.level for node in graph.node_ids.values()) == [0, 1, 2]
    assert all(
        node.reaction.highlight == "normal"
        for node in graph.node_ids.values()
        if node.reaction is not None
    )

    route_planner.commit_route_plan_to_graph(result, "plan_2", experiment)

    assert result.selected_plan_id == "plan_2"
    assert result.candidate_routes[0].materialized is True
    assert result.candidate_routes[1].materialized is True
