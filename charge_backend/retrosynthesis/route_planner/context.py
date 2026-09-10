###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

from pydantic import BaseModel, Field


class RouteContextItem(BaseModel):
    route_number: int
    text: str


class RouteContext(BaseModel):
    items: list[RouteContextItem] = Field(default_factory=list)

    def format(self) -> str:
        lines = ["Candidate retrosynthesis routes:"]
        for item in self.items:
            lines.extend(["", item.text])
        return "\n".join(lines)

    def format_selected(self, route_numbers: list[int]) -> str:
        selected = [item for item in self.items if item.route_number in route_numbers]
        return RouteContext(items=selected or self.items).format()


def build_route_context(summarized_routes, limit=10) -> RouteContext:
    return RouteContext(
        items=[
            RouteContextItem(
                route_number=index,
                text=_route_context_item_text(index, route),
            )
            for index, route in enumerate(summarized_routes[:limit], start=1)
        ]
    )


def route_context_for_prompt(summarized_routes, limit=10):
    return build_route_context(summarized_routes, limit).format()


def _route_context_item_text(index: int, route) -> str:
    summary = getattr(route, "summary", None)
    text = getattr(summary, "text", "") or getattr(summary, "summary", "")
    error = getattr(summary, "error", "")

    candidate = route.candidate
    contracted_unique_reactions = getattr(candidate, "contracted_unique_reactions", None)
    route_depth = getattr(candidate, "route_depth", None)
    average_template_occurrence = getattr(
        candidate,
        "average_template_occurrence",
        None,
    )
    average_policy_probability = getattr(
        candidate,
        "average_policy_probability",
        None,
    )
    rank_score = getattr(candidate, "rank_score", None)
    buyable_leaf_count = getattr(candidate, "buyable_leaf_count", None)
    total_leaf_count = getattr(candidate, "total_leaf_count", None)
    buyable_leaf_ratio = getattr(candidate, "buyable_leaf_ratio", None)
    route_steps = getattr(candidate, "route_steps", None) or []

    lines = [f"Route {index}:"]
    if rank_score is not None:
        lines.append(f"- rank_score: {rank_score:.3f}")
    if buyable_leaf_count is not None and total_leaf_count is not None:
        leaf_text = f"{buyable_leaf_count}/{total_leaf_count}"
        if buyable_leaf_ratio is not None:
            leaf_text += f" ({buyable_leaf_ratio:.2f})"
        lines.append(f"- buyable_leaves: {leaf_text}")
    if contracted_unique_reactions is not None:
        lines.append(f"- contracted_unique_reactions: {contracted_unique_reactions}")
    if route_depth is not None:
        lines.append(f"- route_depth: {route_depth}")
    if average_template_occurrence is not None:
        lines.append(f"- average_template_occurrence: {average_template_occurrence}")
    if average_policy_probability is not None:
        lines.append(
            f"- average_policy_probability (mean template model confidence): {average_policy_probability}"
        )
    if text:
        lines.append(f"- summary: {text}")
    if error:
        lines.append(f"- summary_error: {error}")
    if route_steps:
        lines.append("- route_steps:")
        for step in route_steps:
            precursors = " + ".join(
                str(smiles) for smiles in step.get("precursor_smiles", [])
            )
            lines.append(
                "  - "
                f"{step.get('product_smiles')} <= {precursors}; "
                f"template_occurrence={step.get('template_occurrence')}"
            )
    return "\n".join(lines)
