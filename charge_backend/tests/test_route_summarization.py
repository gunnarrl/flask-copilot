###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

import asyncio

from charge_backend.retrosynthesis import route_planner


def test_summarization_sends_contracted_route_and_patent_evidence_to_agent(
    monkeypatch,
):
    route = {
        "type": "mol",
        "smiles": "P",
        "children": [
            {
                "type": "reaction",
                "metadata": {"template": "t", "library_occurrence": 12},
                "children": [{"type": "mol", "smiles": "A", "in_stock": True}],
            }
        ],
    }
    candidate = route_planner.make_route_candidate("P", route)
    evidence = route_planner.PatentEvidence(
        source_id="US999",
        matched_precursors=[{"role": "Reactant", "name": "A", "smiles": "A"}],
        product_components=[{"role": "Product", "name": "P", "smiles": "P"}],
        other_components=[{"role": "Agent", "name": "acid"}],
        reported_yield="82",
        excerpt="A was converted to P.",
    )
    monkeypatch.setattr(
        "charge_backend.retrosynthesis.route_planner.summarization.find_step_evidence",
        lambda product, precursors: [evidence],
    )

    class Agent:
        async def run(self):
            return "Supported route summary"

    class Experiment:
        def create_agent_with_experiment_state(self, task):
            self.task = task
            return Agent()

    experiment = Experiment()
    result = asyncio.run(route_planner.summarize_routes([candidate], experiment))

    assert result[0].summary.summary == "Supported route summary"
    assert result[0].summary.route_id == candidate.route_id
    assert "Contracted unique reactions: 1" in experiment.task.user_prompt
    assert "Source: US999" in experiment.task.user_prompt
    assert candidate.route_id not in experiment.task.user_prompt
