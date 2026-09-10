###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

from charge_backend.retrosynthesis import route_planner


def test_route_candidate_contracts_consecutive_matching_templates(monkeypatch):
    monkeypatch.setattr(
        "charge_backend.retrosynthesis.route_planner.ranking.is_purchasable",
        lambda smiles: ["catalog"],
    )
    route = {
        "type": "mol",
        "smiles": "P",
        "children": [
            {
                "type": "reaction",
                "metadata": {"template": "same", "library_occurrence": 8},
                "children": [
                    {
                        "type": "mol",
                        "smiles": "A",
                        "children": [
                            {
                                "type": "reaction",
                                "metadata": {
                                    "template": "same",
                                    "library_occurrence": 4,
                                },
                                "children": [
                                    {"type": "mol", "smiles": "B", "in_stock": True}
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }

    candidate = route_planner.make_route_candidate("P", route)

    assert candidate.reaction_count == 2
    assert candidate.contracted_unique_reactions == 1
    assert candidate.route_steps[0]["precursor_smiles"] == ["B"]
    assert candidate.route_steps[0]["contracted_reaction_count"] == 2
    assert candidate.buyable_leaf_ratio == 1.0


def test_route_candidate_does_not_contract_differently_mapped_templates(monkeypatch):
    monkeypatch.setattr(
        "charge_backend.retrosynthesis.route_planner.ranking.is_purchasable",
        lambda smiles: ["catalog"],
    )
    route = {
        "type": "mol",
        "smiles": "CCO",
        "children": [
            {
                "type": "reaction",
                "metadata": {"template": "[C:1]-[O:2]"},
                "children": [
                    {
                        "type": "mol",
                        "smiles": "CC=O",
                        "children": [
                            {
                                "type": "reaction",
                                "metadata": {"template": "[C:2]-[O:1]"},
                                "children": [
                                    {"type": "mol", "smiles": "CC"},
                                    {"type": "mol", "smiles": "O"},
                                ],
                            }
                        ],
                    },
                    {"type": "mol", "smiles": "O"},
                ],
            }
        ],
    }

    candidate = route_planner.make_route_candidate("CCO", route)

    assert candidate.contracted_unique_reactions == 2


def test_ranking_and_diversity_keep_the_best_distinct_routes():
    def candidate(route_id, *, buyable, length, occurrence):
        return route_planner.RouteCandidate(
            route_id=route_id,
            target_smiles="P",
            route={},
            all_leaves_in_stock=buyable == 1,
            reaction_count=length,
            contracted_unique_reactions=length,
            route_depth=length,
            average_template_occurrence=occurrence,
            buyable_leaf_ratio=buyable,
        )

    routes = [
        candidate("best", buyable=1, length=1, occurrence=10),
        candidate("similar", buyable=0.8, length=2, occurrence=8),
        candidate("distinct", buyable=0.5, length=3, occurrence=2),
    ]
    ranked = route_planner.rank_routes(routes)
    selected = route_planner.select_diverse_routes(
        ranked,
        limit=2,
        similarity_function=lambda first, second: (
            0.9 if {first.route_id, second.route_id} == {"best", "similar"} else 0.1
        ),
    )

    assert ranked[0].route_id == "best"
    assert [route.route_id for route in selected] == ["best", "distinct"]
    assert all(route.rank_score == 0 for route in routes)
