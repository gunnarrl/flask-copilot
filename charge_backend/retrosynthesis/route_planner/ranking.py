###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any

from flask_tools.chemistry.smiles_utils import canonicalize_smiles
from charge_backend.moleculedb.purchasable import is_purchasable
from rxnutils.routes.comparison import simple_route_similarity
from rxnutils.routes.readers import read_aizynthfinder_dict


OCCURRENCE_KEYS = (
    "library_occurence",
    "library_occurrence",
    "template_occurrence",
    "template_occurence",
)


@dataclass(frozen=True)
class RouteCandidate:
    route_id: str
    target_smiles: str
    route: dict[str, Any]
    all_leaves_in_stock: bool
    reaction_count: int
    contracted_unique_reactions: int
    route_depth: int
    average_template_occurrence: float
    average_policy_probability: float = 0.0
    buyable_leaf_count: int = 0
    total_leaf_count: int = 0
    buyable_leaf_ratio: float = 0.0
    rank_score: float = 0.0
    route_steps: list[dict[str, Any]] = field(default_factory=list)


def rank_routes(routes) -> list[RouteCandidate]:
    """Score route candidates and return them in descending rank order."""
    scored_routes = score_routes(list(routes))
    return sorted(scored_routes, key=lambda route: (-route.rank_score, route.route_id))


def score_routes(routes: list[RouteCandidate]) -> list[RouteCandidate]:
    """Assign a normalized practicality and evidence score to each route."""
    length_scores = _normalized_scores(
        routes, lambda route: route.contracted_unique_reactions, higher_is_better=False
    )
    depth_scores = _normalized_scores(
        routes, lambda route: route.route_depth, higher_is_better=False
    )
    occurrence_scores = _normalized_scores(
        routes,
        lambda route: route.average_template_occurrence,
        higher_is_better=True,
    )
    policy_scores = _normalized_scores(
        routes,
        lambda route: route.average_policy_probability,
        higher_is_better=True,
    )

    scored = []
    for index, route in enumerate(routes):
        rank_score = (
            0.40 * route.buyable_leaf_ratio
            + 0.20 * length_scores[index]
            + 0.15 * depth_scores[index]
            + 0.15 * occurrence_scores[index]
            + 0.10 * policy_scores[index]
        )
        scored.append(replace(route, rank_score=rank_score))
    return scored


def _normalized_scores(routes, value_func, higher_is_better: bool) -> list[float]:
    values = [float(value_func(route)) for route in routes]
    if not values:
        return []
    lowest = min(values)
    highest = max(values)
    if lowest == highest == 0.0:
        return [0.0 for _ in values]
    if lowest == highest:
        return [1.0 for _ in values]
    if higher_is_better:
        return [(value - lowest) / (highest - lowest) for value in values]
    return [(highest - value) / (highest - lowest) for value in values]


def select_diverse_routes(
    ranked_routes,
    limit=10,
    similarity_threshold=0.8,
    similarity_function=None,
):
    """Select high-ranking routes while preferring dissimilar alternatives."""
    if not similarity_function:
        return ranked_routes[:limit]

    selected = []
    rejected = []

    for route in ranked_routes:
        if not selected:
            selected.append(route)
        else:
            max_sim = 0
            for s in selected:
                max_sim = max(max_sim, similarity_function(s, route))
            if max_sim < similarity_threshold:
                selected.append(route)
            else:
                rejected.append(route)
        if len(selected) == limit:
            break

    while len(selected) < limit:
        if len(rejected) != 0:
            selected.append(rejected.pop(0))
        else:
            break

    return selected


def prefer_unique_route_signatures(ranked_routes: list[RouteCandidate]) -> list[RouteCandidate]:
    """Move duplicate route signatures behind unique routes without dropping them."""
    unique = []
    duplicates = []
    seen = set()
    for route in ranked_routes:
        signature = route_signature(route.route_steps)
        if signature in seen:
            duplicates.append(route)
        else:
            seen.add(signature)
            unique.append(route)
    return unique + duplicates


def aizynth_route_similarity(first: RouteCandidate, second: RouteCandidate) -> float:
    """Calculate AiZynthFinder route similarity between two candidates."""
    routes = [
        read_aizynthfinder_dict(first.route),
        read_aizynthfinder_dict(second.route),
    ]
    return float(simple_route_similarity(routes)[0, 1])


def route_id_for_route(route: dict[str, Any]) -> str:
    """Return a stable content-derived identifier for a route tree."""
    route_json = json.dumps(route, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(route_json.encode("utf-8")).hexdigest()


def reaction_count(route: dict[str, Any]) -> int:
    """Count reaction nodes in a route tree."""
    count = 1 if route.get("type") == "reaction" else 0
    for child in route.get("children", []):
        count += reaction_count(child)
    return count


def route_depth(route: dict[str, Any]) -> int:
    """Return the maximum number of reaction levels in a route tree."""
    deepest_child = 0
    for child in route.get("children", []):
        deepest_child = max(deepest_child, route_depth(child))

    if route.get("type") == "reaction":
        return deepest_child + 1
    return deepest_child


def all_leaves_in_stock(route: dict[str, Any]) -> bool:
    """Return whether every molecular leaf is purchasable."""
    leaves = route_leaves(route)
    return bool(leaves) and leaf_buyable_count(leaves) == len(leaves)


def leaf_buyable_count(leaves: list[dict[str, Any]]) -> int:
    """Count purchasable molecular leaves."""
    count = 0
    for leaf in leaves:
        smiles = leaf.get("smiles")
        if isinstance(smiles, str) and is_purchasable(smiles):
            count += 1
    return count


def route_leaves(route: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect terminal molecule nodes from a route tree."""
    if route.get("type") == "mol" and not route.get("children"):
        return [route]

    leaves = []
    for child in route.get("children", []):
        leaves.extend(route_leaves(child))
    return leaves


def template_occurrence(reaction: dict[str, Any]) -> int | None:
    """Read a template occurrence count from reaction metadata."""
    metadata = reaction.get("metadata") or {}
    for key in OCCURRENCE_KEYS:
        value = metadata.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def reaction_template(reaction: dict[str, Any] | None) -> str | None:
    """Read the SMARTS template associated with a reaction node."""
    if reaction is None:
        return None
    metadata = reaction.get("metadata") or {}
    value = metadata.get("template") or metadata.get("smarts")
    return str(value) if value else None


def _normalize_template_string(template: str) -> str:
    return template.replace(" ", "")


def normalized_reaction_template(reaction: dict[str, Any] | None) -> str | None:
    """Return the normalized SMARTS template for a reaction node."""
    template = reaction_template(reaction)
    if not template:
        return None
    return _normalize_template_string(template)


def extract_template_occurrences(route: dict[str, Any]) -> list[int]:
    """Collect template occurrence counts from a route tree."""
    occurrences = []

    if route.get("type") == "reaction":
        occurrence = template_occurrence(route)
        if occurrence is not None:
            occurrences.append(occurrence)

    for child in route.get("children", []):
        occurrences.extend(extract_template_occurrences(child))

    return occurrences


def extract_policy_probabilities(route: dict[str, Any]) -> list[float]:
    """Collect valid policy probabilities from a route tree."""
    probabilities = []
    if route.get("type") == "reaction":
        value = (route.get("metadata") or {}).get("policy_probability")
        try:
            probabilities.append(float(value))
        except (TypeError, ValueError):
            pass

    for child in route.get("children", []):
        probabilities.extend(extract_policy_probabilities(child))
    return probabilities


def first_reaction_child(molecule: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first reaction child of a molecule node, if present."""
    for child in molecule.get("children", []):
        if child.get("type") == "reaction":
            return child
    return None


def _has_shared_direct_precursor(
    parent_precursors: list[dict[str, Any]],
    child_molecule: dict[str, Any],
    child_reaction: dict[str, Any],
) -> bool:
    parent_keys = {
        canonicalize_smiles(precursor.get("smiles"))
        for precursor in parent_precursors
        if precursor is not child_molecule
    }
    child_keys = {
        canonicalize_smiles(precursor.get("smiles"))
        for precursor in child_reaction.get("children", [])
        if precursor.get("type") == "mol"
    }
    parent_keys.discard("")
    child_keys.discard("")
    return bool(parent_keys & child_keys)


def _should_contract_child_reaction(
    parent_reaction: dict[str, Any],
    parent_precursors: list[dict[str, Any]],
    child_molecule: dict[str, Any],
) -> bool:
    child_reaction = first_reaction_child(child_molecule)
    parent_template = reaction_template(parent_reaction)
    child_template = reaction_template(child_reaction)
    if child_reaction is None or not parent_template or not child_template:
        return False
    if parent_template == child_template:
        return True
        
    # this could be removed if shared-precursor contraction is reliable without requiring matching template shape.
    same_template_shape = normalized_reaction_template(
        parent_reaction
    ) == normalized_reaction_template(child_reaction)
    return same_template_shape and _has_shared_direct_precursor(
        parent_precursors, child_molecule, child_reaction
    )


def _expanded_precursors_for_reaction(
    molecule: dict[str, Any],
    parent_reaction: dict[str, Any],
    parent_precursors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reaction = first_reaction_child(molecule)
    if reaction is None or not _should_contract_child_reaction(
        parent_reaction,
        parent_precursors,
        molecule,
    ):
        return [molecule]

    expanded = []
    reaction_precursors = [
        item for item in reaction.get("children", []) if item.get("type") == "mol"
    ]
    for child in reaction.get("children", []):
        if child.get("type") == "mol":
            expanded.extend(
                _expanded_precursors_for_reaction(
                    child,
                    reaction,
                    reaction_precursors,
                )
            )
    return expanded


def add_contracted_route_steps(
    molecule: dict[str, Any],
    depth: int,
    steps: list[dict[str, Any]],
    starting_materials: list[dict[str, Any]],
) -> None:
    """Append contracted reaction steps and starting materials for a route branch."""
    reaction = first_reaction_child(molecule)
    if reaction is None:
        starting_materials.append(
            {
                "smiles": molecule.get("smiles"),
                "in_stock": molecule.get("in_stock"),
            }
        )
        return

    template = reaction_template(reaction)
    direct_precursors = [
        child for child in reaction.get("children", []) if child.get("type") == "mol"
    ]
    contracted_precursors = []
    for precursor in direct_precursors:
        contracted_precursors.extend(
            _expanded_precursors_for_reaction(
                precursor,
                reaction,
                direct_precursors,
            )
        )

    steps.append(
        {
            "depth": depth,
            "product_smiles": molecule.get("smiles"),
            "precursor_smiles": [
                precursor.get("smiles") for precursor in contracted_precursors
            ],
            "template": template,
            "template_occurrence": template_occurrence(reaction),
            "contracted_reaction_count": max(
                1,
                reaction_count(reaction)
                - sum(reaction_count(precursor) for precursor in contracted_precursors),
            ),
        }
    )

    for precursor in contracted_precursors:
        add_contracted_route_steps(
            precursor,
            depth=depth + 1,
            steps=steps,
            starting_materials=starting_materials,
        )


def contracted_route_summary(route: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return contracted reaction steps and starting materials for a route."""
    steps = []
    starting_materials = []
    add_contracted_route_steps(
        route,
        depth=0,
        steps=steps,
        starting_materials=starting_materials,
    )
    return steps, starting_materials


def route_signature(route_steps: list[dict[str, Any]]) -> str:
    """Build a stable signature from normalized route steps."""
    signature = []
    for step in route_steps:
        signature.append(
            {
                "product": canonicalize_smiles(step.get("product_smiles")),
                "precursors": [
                    canonicalize_smiles(smiles)
                    for smiles in step.get("precursor_smiles", [])
                ],
                "template": _normalize_template_string(step.get("template") or ""),
            }
        )
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def make_route_candidate(target_smiles: str, route: dict[str, Any]) -> RouteCandidate:
    """Create a ranked-route candidate and calculate its route metrics."""
    occurrences = extract_template_occurrences(route)
    average_occurrence = sum(occurrences) / len(occurrences) if occurrences else 0.0
    probabilities = extract_policy_probabilities(route)
    average_probability = (
        sum(probabilities) / len(probabilities) if probabilities else 0.0
    )

    route_steps, starting_materials = contracted_route_summary(route)
    total_leaf_count = len(starting_materials)
    buyable_leaf_count = leaf_buyable_count(starting_materials)
    buyable_leaf_ratio = (
        buyable_leaf_count / total_leaf_count if total_leaf_count else 0.0
    )

    return RouteCandidate(
        route_id=route_id_for_route(route),
        target_smiles=target_smiles,
        route=route,
        all_leaves_in_stock=bool(total_leaf_count)
        and buyable_leaf_count == total_leaf_count,
        reaction_count=reaction_count(route),
        contracted_unique_reactions=len(route_steps),
        route_steps=route_steps,
        route_depth=route_depth(route),
        average_template_occurrence=average_occurrence,
        average_policy_probability=average_probability,
        buyable_leaf_count=buyable_leaf_count,
        total_leaf_count=total_leaf_count,
        buyable_leaf_ratio=buyable_leaf_ratio,
    )


def route_candidate_to_dict(candidate: RouteCandidate) -> dict[str, Any]:
    """Serialize a route candidate to a plain dictionary."""
    return {
        "route_id": candidate.route_id,
        "target_smiles": candidate.target_smiles,
        "route": candidate.route,
        "all_leaves_in_stock": candidate.all_leaves_in_stock,
        "reaction_count": candidate.reaction_count,
        "contracted_unique_reactions": candidate.contracted_unique_reactions,
        "route_steps": candidate.route_steps,
        "route_depth": candidate.route_depth,
        "average_template_occurrence": candidate.average_template_occurrence,
        "average_policy_probability": candidate.average_policy_probability,
        "buyable_leaf_count": candidate.buyable_leaf_count,
        "total_leaf_count": candidate.total_leaf_count,
        "buyable_leaf_ratio": candidate.buyable_leaf_ratio,
        "rank_score": candidate.rank_score,
    }
