###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

import argparse
import asyncio
import json
from typing import Any

from agent_framework import AgentSession, Message
from charge.clients.agent import Agent, AgentBackend, AgentRuntimeConfig
from charge.experiments.experiment import AgentRegistryEntry
from charge.tasks.task import Task
from charge_backend.backend_manager import FlaskActionManager
from charge_backend.retrosynthesis import route_planner


class FakeWebSocket:
    headers = {}

    def __init__(self):
        self.messages = []
        self.last_payload = None

    async def send_json(self, payload):
        self.messages.append(payload)
        self.last_payload = payload


class FakeAgent(Agent):
    def __init__(self, task: Task | None, memory: str = ""):
        super().__init__(task)
        self.memory = memory

    def run(self, **kwargs) -> str:
        return ""

    def load_memory(self, json_str: str) -> None:
        self.memory = json_str

    def save_memory(self) -> str:
        return self.memory

    def get_model_info(self) -> dict[str, Any]:
        return {"backend": "dummy", "model": "dummy-model"}


class FakeBackend(AgentBackend):
    def __init__(self):
        super().__init__(
            model="dummy-model",
            api_key=None,
            base_url=None,
            reasoning_effort=None,
            model_kwargs=None,
            backend="dummy",
        )

    def create_agent(self, task, **kwargs):
        return FakeAgent(task)


def make_manager() -> FlaskActionManager:
    return FlaskActionManager(
        websocket=FakeWebSocket(),
        args=argparse.Namespace(backend="openai", model="gpt-4o-mini"),
        username="test-user",
    )


def set_planner_memory(manager: FlaskActionManager, memory: str) -> None:
    manager.experiment.agent_registry["route-planning:planner"] = AgentRegistryEntry(
        agent=FakeAgent(task=None, memory=memory),
        runtime_config=AgentRuntimeConfig(backend="dummy", model="dummy-model"),
    )


def planner_memory(manager: FlaskActionManager) -> str:
    agent = manager.experiment.agent_registry["route-planning:planner"].agent
    return agent.save_memory()


def set_planner_message(manager: FlaskActionManager, text: str) -> None:
    session = AgentSession()
    session.state["in_memory"] = {
        "messages": [Message(role="assistant", contents=[text])]
    }
    set_planner_memory(manager, json.dumps(session.to_dict()))


def memory_text(memory: str) -> str:
    session = AgentSession.from_dict(json.loads(memory))
    return "\n".join(message.text for message in session.state["in_memory"]["messages"])


def planner_text(manager: FlaskActionManager) -> str:
    return memory_text(planner_memory(manager))


def route_procedure_step() -> route_planner.RouteProcedureStep:
    return route_planner.RouteProcedureStep(
        step_label="Step 1",
        product_smiles="CCO",
        precursor_smiles=["CC=O"],
        description="Disconnect the target.",
        transformation_type="main bond formation",
        evidence_level="template",
        evidence_basis="Template route evidence.",
        source_refs=["Route 1"],
        conditions_summary="Template-only; conditions not provided.",
        yield_summary="No yield reported.",
        confidence="medium",
        main_uncertainty="Forward conditions need verification.",
    )


def make_plan(plan_id: str, title: str) -> route_planner.CandidateRoutePlan:
    return route_planner.CandidateRoutePlan(
        plan_id=plan_id,
        plan=route_planner.RoutePlanContent(
            title=title,
            route_type="template_based",
            route_strategy="Use the supported route.",
            route_value="Tests route-planning branch behavior.",
            evidence_overview="Template route evidence.",
            novelty_rationale="Fixture route; no novelty.",
            procedure_steps=[route_procedure_step()],
        ),
    )


def test_plan_branch_compaction_does_not_leak_between_plans():
    async def run() -> None:
        manager = make_manager()
        manager.experiment.backend = FakeBackend()
        set_planner_message(manager, "initial five plans")
        result = route_planner.RoutePlanningResult(
            target_smiles="CCO",
            candidate_routes=[
                make_plan(f"plan_{index}", f"Plan {index}") for index in range(1, 6)
            ],
        )
        result.base_branch_state = route_planner.save_route_planning_branch_state(
            manager.experiment
        )
        manager.experiment.route_planning_result = result

        async def question_plan_1():
            assert "initial five plans" in planner_text(manager)
            assert len(manager.experiment.route_planning_result.candidate_routes) == 5

        await manager._run_in_route_plan_branch(
            result,
            "plan_1",
            question_plan_1,
            latest_user_message="Plan 1 question",
            latest_assistant_message="Plan 1 answer",
        )

        async def refine_plan_1():
            text = planner_text(manager)
            assert "Plan 1 question" in text
            assert "Plan 1 answer" in text

        await manager._run_in_route_plan_branch(
            result,
            "plan_1",
            refine_plan_1,
            latest_user_message="Refine plan 1",
            latest_assistant_message="Plan 1 refined",
        )

        async def question_plan_2():
            text = planner_text(manager)
            assert "initial five plans" in text
            assert "Plan 1 question" not in text

        await manager._run_in_route_plan_branch(
            result,
            "plan_2",
            question_plan_2,
            latest_user_message="Plan 2 question",
            latest_assistant_message="Plan 2 answer",
        )

        plan_1_memory = result.candidate_routes[0].branch_state["agentSessions"][
            "route-planning:planner"
        ]["memory"]
        plan_2_memory = result.candidate_routes[1].branch_state["agentSessions"][
            "route-planning:planner"
        ]["memory"]
        plan_1_text = memory_text(plan_1_memory)
        plan_2_text = memory_text(plan_2_memory)
        assert "Refine plan 1" in plan_1_text
        assert "Plan 1 refined" in plan_1_text
        assert "Plan 2 question" not in plan_1_text
        assert "Plan 2 question" in plan_2_text
        assert "Plan 2 answer" in plan_2_text
        assert "Refine plan 1" not in plan_2_text

    asyncio.run(run())


def test_saving_compact_branch_updates_the_live_planner_session():
    manager = make_manager()
    manager.experiment.backend = FakeBackend()
    session = AgentSession()
    session.state["in_memory"] = {
        "messages": [Message(role="user", contents=["large raw history " * 1000])]
    }
    raw_memory = json.dumps(session.to_dict())
    set_planner_memory(manager, raw_memory)
    planner = manager.experiment.agent_registry["route-planning:planner"].agent
    other_agent = FakeAgent(task=None, memory="unrelated memory")
    manager.experiment.agent_registry["custom:main"] = AgentRegistryEntry(
        agent=other_agent,
        runtime_config=AgentRuntimeConfig(backend="dummy", model="dummy-model"),
    )
    result = route_planner.RoutePlanningResult(
        target_smiles="CCO",
        candidate_routes=[make_plan("plan_1", "Plan 1")],
    )

    branch_state = route_planner.save_compact_route_planning_branch_state(
        manager.experiment,
        result,
        latest_user_message="Why this route?",
        latest_assistant_message="It uses the strongest evidence.",
    )

    saved_memory = branch_state["agentSessions"]["route-planning:planner"]["memory"]
    assert manager.experiment.agent_registry["route-planning:planner"].agent is planner
    assert manager.experiment.agent_registry["custom:main"].agent is other_agent
    assert planner_memory(manager) == saved_memory
    assert len(saved_memory) < len(raw_memory)
    restored = AgentSession.from_dict(json.loads(saved_memory))
    messages = restored.state["in_memory"]["messages"]
    assert all(isinstance(message, Message) for message in messages)
    combined = "\n".join(message.text for message in messages)
    assert "large raw history" not in combined
    assert "Why this route?" in combined
    assert "It uses the strongest evidence." in combined

    set_planner_memory(manager, "changed after save")
    route_planner.restore_route_planning_branch_state(manager.experiment, branch_state, result)
    assert planner_memory(manager) == saved_memory


def test_route_planning_more_uses_and_updates_shared_base_branch(monkeypatch):
    async def run() -> None:
        manager = make_manager()
        manager.experiment.backend = FakeBackend()
        set_planner_message(manager, "plan 1 private memory")
        plan_1_branch = route_planner.save_route_planning_branch_state(manager.experiment)
        set_planner_message(manager, "initial five plans")
        result = route_planner.RoutePlanningResult(
            target_smiles="CCO",
            candidate_routes=[
                make_plan(f"plan_{index}", f"Plan {index}") for index in range(1, 6)
            ],
        )
        result.base_branch_state = route_planner.save_route_planning_branch_state(
            manager.experiment
        )
        result.candidate_routes[0].branch_state = plan_1_branch
        manager.experiment.route_planning_result = result
        manager.route_planner_tool_runtime = lambda: None

        async def fake_plan_more(
            result, experiment, tool_runtime, user_feedback, callback=None
        ):
            del tool_runtime, callback
            assert "initial five plans" in planner_text(manager)
            assert user_feedback == "Avoid route 1."
            result.candidate_routes.append(make_plan("plan_6", "Plan 6"))
            return object()

        monkeypatch.setattr(route_planner, "plan_more_candidate_routes", fake_plan_more)

        await manager.handle_route_planning_more(
            {
                "query": "Avoid route 1.",
                "chatAgentKey": "route-planning:more",
            }
        )

        assert len(result.candidate_routes) == 6
        assert result.candidate_routes[0].branch_state == plan_1_branch
        assert result.candidate_routes[5].branch_state is None
        base_memory = result.base_branch_state["agentSessions"]["route-planning:planner"][
            "memory"
        ]
        assert "Avoid route 1." in memory_text(base_memory)
        assert "Generated additional route plans." in memory_text(base_memory)
        response = manager.websocket.messages[-3]
        assert response["type"] == "route-planning-result-response"
        assert "base_branch_state" not in response["result"]
        agent_response = manager.websocket.messages[-2]
        assert agent_response["type"] == "agent-response"
        assert agent_response["agentKey"] == "route-planning:more"
        assert "Avoid route 1." in memory_text(agent_response["agent"]["memory"])

    asyncio.run(run())


def test_route_planning_select_restores_branch_and_returns_graph(monkeypatch):
    async def run() -> None:
        manager = make_manager()
        manager.experiment.backend = FakeBackend()
        set_planner_message(manager, "plan 1 private memory")
        plan_1_branch = route_planner.save_route_planning_branch_state(manager.experiment)
        set_planner_message(manager, "initial five plans")
        plan = make_plan("plan_1", "Plan 1")
        plan.route_steps = [
            route_planner.RouteStep(
                step_id="make_target",
                product_smiles="CCO",
                precursor_smiles=["CC=O"],
            )
        ]
        result = route_planner.RoutePlanningResult(
            target_smiles="CCO",
            candidate_routes=[plan, make_plan("plan_2", "Plan 2")],
        )
        result.base_branch_state = route_planner.save_route_planning_branch_state(
            manager.experiment
        )
        result.candidate_routes[0].branch_state = plan_1_branch
        manager.experiment.route_planning_result = result

        commit_route_plan = route_planner.commit_route_plan_to_graph

        def commit_after_restore(*args, **kwargs):
            assert "plan 1 private memory" in planner_text(manager)
            return commit_route_plan(*args, **kwargs)

        monkeypatch.setattr(
            route_planner,
            "commit_route_plan_to_graph",
            commit_after_restore,
        )

        await manager.handle_route_planning_select({"planId": "plan_1"})

        assert result.selected_plan_id == "plan_1"
        assert result.candidate_routes[0].branch_state is not None
        response = manager.websocket.messages[-2]
        assert response["type"] == "route-planning-select-response"
        assert response["planId"] == "plan_1"
        assert set(response["graphContext"]["node_ids"]) == {
            "plan_1_node_0",
            "plan_1_node_1",
        }
        assert "base_branch_state" not in response["result"]
        assert "branch_state" not in response["result"]["candidate_routes"][0]
        assert manager.websocket.messages[-1] == {"type": "complete"}

    asyncio.run(run())
