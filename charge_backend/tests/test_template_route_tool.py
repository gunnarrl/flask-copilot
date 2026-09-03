import asyncio
from types import SimpleNamespace

from charge_backend.builtin_tools import list_builtin_tool_definitions
from charge_backend.retrosynthesis import route_planner


def template_route_tool(monkeypatch, candidates):
    callback = object()

    async def enumerate_routes(config_file, smiles, k):
        assert (config_file, smiles, k) == ("config.yml", "P", 2)
        return candidates

    async def summarize_routes(
        routes, experiment, limit, concurrency, callback: object
    ):
        assert routes == candidates
        assert callback is experiment.agent_registry["route-planning:planner"].agent.callback
        return [
            route_planner.SummarizedRoute(
                candidate=route,
                summary=route_planner.RouteSummary(
                    route.route_id, "Patent-supported route."
                ),
            )
            for route in routes
        ]

    monkeypatch.setattr(
        "charge_backend.builtin_tools.enumerate_template_route_candidates",
        enumerate_routes,
    )
    monkeypatch.setattr(
        "charge_backend.builtin_tools.route_planner.summarize_routes",
        summarize_routes,
    )
    definitions = list_builtin_tool_definitions(
        retrosynthesis_config_file="config.yml",
        experiment=SimpleNamespace(
            agent_registry={
                "route-planning:planner": SimpleNamespace(
                    agent=SimpleNamespace(callback=callback)
                )
            }
        ),
    )
    return next(
        definition.function
        for definition in definitions
        if definition.identifier == "enumerate_template_routes"
    )


def test_template_route_tool_returns_summarized_planner_context(monkeypatch):
    candidate = route_planner.RouteCandidate(
        route_id="route-1",
        target_smiles="P",
        route={"children": ["large raw tree"]},
        all_leaves_in_stock=True,
        reaction_count=1,
        contracted_unique_reactions=1,
        route_depth=1,
        average_template_occurrence=5,
        route_steps=[
            {
                "product_smiles": "P",
                "precursor_smiles": ["A"],
                "template_occurrence": 5,
            }
        ],
    )

    result = asyncio.run(template_route_tool(monkeypatch, [candidate])("P", k=2))

    assert "Route 1:" in result
    assert "Patent-supported route." in result
    assert "P <= A; template_occurrence=5" in result
    assert "large raw tree" not in result


def test_template_route_tool_reports_no_routes(monkeypatch):
    result = asyncio.run(template_route_tool(monkeypatch, [])("P", k=2))

    assert result == "No template routes found for P."
