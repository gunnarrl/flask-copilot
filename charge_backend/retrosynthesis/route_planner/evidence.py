###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass
class PatentEvidence:
    source_id: str
    matched_precursors: list[dict[str, Any]]
    product_components: list[dict[str, Any]]
    other_components: list[dict[str, Any]]
    reported_yield: str | None
    excerpt: str


def _smiles_to_inchi(smiles: str) -> str | None:
    from charge_backend.retrosynthesis.database import smiles_to_inchi

    return smiles_to_inchi(smiles)


def _component_inchi(component: dict[str, Any]) -> str | None:
    inchi = component.get("inchi")
    if isinstance(inchi, str) and inchi:
        return inchi
    smiles = component.get("smiles")
    if isinstance(smiles, str) and smiles:
        return _smiles_to_inchi(smiles)
    return None


def find_step_evidence(
    product: str,
    precursors: list[str],
    limit: int = 3,
) -> list[PatentEvidence]:
    """Find patent evidence matching a product and all requested precursors."""
    from charge_backend.retrosynthesis.database import get_reaction_database

    product_inchi = _smiles_to_inchi(product)

    if not product_inchi:
        return []

    database = get_reaction_database()
    if database is None:
        return []
    database_handle, parser = database

    entries = database_handle.get(product_inchi)
    if not entries:
        return []

    parsed_entries = [parser(product_inchi, entry) for entry in entries]
    return evidence_from_entries(parsed_entries, precursors, limit)


def evidence_from_entries(
    entries: list[Any],
    precursors: list[str],
    limit: int = 3,
) -> list[PatentEvidence]:
    """Extract matching patent evidence from parsed reaction entries."""
    res = []
    precursor_counts = Counter(
        inchi
        for precursor in precursors
        if (inchi := _smiles_to_inchi(precursor))
    )
    if len(precursor_counts) != len(precursors):
        return []

    for entry in entries:
        matched_components = []
        matched_component_ids = set()
        remaining_precursors = precursor_counts.copy()
        for comp in entry.components:
            role = comp.get("role")
            if role in {"Reactant", "Agent"}:
                comp_inchi = _component_inchi(comp)
                if comp_inchi and remaining_precursors[comp_inchi] > 0:
                    matched_components.append(comp)
                    matched_component_ids.add(id(comp))
                    remaining_precursors[comp_inchi] -= 1

        if any(count > 0 for count in remaining_precursors.values()):
            continue

        res.append(
            PatentEvidence(
                source_id=entry.name,
                excerpt=entry.text,
                matched_precursors=matched_components,
                product_components=[
                    comp
                    for comp in entry.components
                    if comp.get("role") == "Product"
                ],
                other_components=[
                    comp
                    for comp in entry.components
                    if comp.get("role") != "Product"
                    and id(comp) not in matched_component_ids
                ],
                reported_yield=(
                    f"{entry.reaction_yield:g}"
                    if entry.reaction_yield >= 0
                    else None
                ),
            )
        )

    return sorted(
        res,
        key=lambda entry: (
            -_reported_yield_sort_value(entry.reported_yield),
            entry.source_id,
        ),
    )[:limit]


def _reported_yield_sort_value(value: str | None) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0
