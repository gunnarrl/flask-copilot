from types import SimpleNamespace

from charge_backend.retrosynthesis import route_planner


def test_evidence_matches_all_precursors_and_preserves_patent_details():
    entries = [
        SimpleNamespace(
            name="US20090082592A1_0049",
            text="The mixture was heated.",
            components=[
                {"role": "Reactant", "name": "ethanol", "smiles": "CCO"},
                {"role": "Agent", "name": "water", "smiles": "O"},
                {
                    "role": "Product",
                    "name": "product",
                    "smiles": "CCCl",
                    "quantities": [{"type": "Yield", "value": 75}],
                },
                {"role": "Agent", "name": "acid"},
            ],
            reaction_yield=75.0,
        )
    ]

    evidence = route_planner.evidence_from_entries(entries, ["O", "CCO"])

    assert len(evidence) == 1
    assert evidence[0].source_id == "US20090082592A1_0049"
    assert evidence[0].reported_yield == "75"
    assert [item["name"] for item in evidence[0].matched_precursors] == [
        "ethanol",
        "water",
    ]
    assert evidence[0].other_components == [{"role": "Agent", "name": "acid"}]
