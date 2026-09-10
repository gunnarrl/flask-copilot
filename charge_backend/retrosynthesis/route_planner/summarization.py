###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

import asyncio
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable, TYPE_CHECKING
from uuid import uuid4

from charge.tasks.task import Task

from .evidence import find_step_evidence
from .ranking import (
    RouteCandidate,
    contracted_route_summary,
)

if TYPE_CHECKING:
    from charge.clients.agent import AgentCallbackType
    from charge_backend.flask_experiment import FlaskExperiment


@dataclass(frozen=True)
class RouteSummary:
    route_id: str
    summary: str | None
    error: str | None = None


@dataclass(frozen=True)
class SummarizedRoute:
    candidate: RouteCandidate
    summary: RouteSummary


class RouteSummarizer:
    def __init__(self, experiment: "FlaskExperiment"):
        self.experiment = experiment

    async def summarize(self, candidate: RouteCandidate) -> tuple[RouteSummary, str]:
        """Summarize one route and return its summary and summarizer agent ID."""
        prompt = build_summary_prompt(route_summary_input(candidate))
        agent_id = f"summarizer-{uuid4()}"
        try:
            task = Task(
                system_prompt="You summarize retrosynthesis routes concisely and only use the provided route data.",
                user_prompt=prompt,
            )
            agent = self.experiment.create_agent_with_experiment_state(task=task)
            agent_id = str(getattr(agent, "agent_key", agent_id))
            result = await agent.run()
        except Exception as exc:
            return (
                RouteSummary(
                    route_id=candidate.route_id,
                    summary=None,
                    error=str(exc),
                ),
                agent_id,
            )
        return (
            RouteSummary(route_id=candidate.route_id, summary=str(result).strip()),
            agent_id,
        )


async def summarize_routes(
    candidates: list[RouteCandidate],
    experiment: "FlaskExperiment",
    limit: int = 10,
    concurrency: int = 4,
    callback: "AgentCallbackType" = None,
    status_callback: Callable[[str], Awaitable[None]] | None = None,
) -> list[SummarizedRoute]:
    """Summarize route candidates concurrently while reporting their results."""
    semaphore = asyncio.Semaphore(concurrency)
    summarizer = RouteSummarizer(experiment)
    selected_candidates = candidates[:limit]
    total = len(selected_candidates)

    if status_callback is not None:
        await status_callback(f"Summarization started for {total} routes.")

    async def summarize_one(
        index: int, candidate: RouteCandidate
    ) -> SummarizedRoute:
        async with semaphore:
            summary, agent_id = await summarizer.summarize(candidate)
            if callback is not None:
                call_id = f"summarize_route:{agent_id}:{candidate.route_id}"
                await callback.on_tool_call(
                    "summarize_route",
                    {"route_number": index, "route_id": candidate.route_id},
                    source=agent_id,
                    call_id=call_id,
                )
                if summary.error:
                    await callback.on_tool_result(
                        "summarize_route",
                        summary.error,
                        is_error=True,
                        source=agent_id,
                        call_id=call_id,
                    )
                else:
                    await callback.on_tool_result(
                        "summarize_route",
                        summary.summary,
                        source=agent_id,
                        call_id=call_id,
                    )
            return SummarizedRoute(candidate=candidate, summary=summary)

    summaries = await asyncio.gather(
        *(
            summarize_one(index, candidate)
            for index, candidate in enumerate(selected_candidates, start=1)
        )
    )
    if status_callback is not None:
        await status_callback(f"Summarization completed for {total} routes.")
    return summaries


def route_summary_input(candidate: RouteCandidate) -> dict[str, Any]:
    """Build the evidence packet used to summarize a route candidate."""
    steps, starting_materials = contracted_route_summary(candidate.route)
    for step in steps:
        evidence = find_step_evidence(
            str(step.get("product_smiles") or ""),
            [
                str(precursor)
                for precursor in step.get("precursor_smiles", [])
                if precursor
            ],
        )
        if evidence:
            step["patent_evidence"] = [asdict(item) for item in evidence]

    return {
        "target_smiles": candidate.target_smiles,
        "contracted_unique_reactions": candidate.contracted_unique_reactions,
        "route_depth": candidate.route_depth,
        "average_template_occurrence": candidate.average_template_occurrence,
        "route_steps": steps,
        "starting_materials": starting_materials,
    }


def build_summary_prompt(route_input: dict[str, Any]) -> str:
    """Build the evidence-only route summarization prompt."""
    route_packet = format_route_summary_input(route_input)
    return f"""Summarize this proposed retrosynthesis route using only the provided route data.

Write a detailed route summary in markdown with these sections:
- Route overview
- Step-by-step evidence
- Starting materials
- Patent-backed conditions and yields
- Template-only assumptions
- Practical route judgment

Preserve the route sequence, branch assembly, starting materials,
patent-supported transformations, matched precursors, reagents, solvents,
catalysts, agents, quantities, temperature, time, workup, purification,
characterization, and yield when present. Clearly separate patent evidence from
template-only assumptions. Do not invent reagents, reaction conditions, yields,
mechanisms, patent support, or experimental evidence that is not present.

Route evidence:
{route_packet}
"""


def format_route_summary_input(route_input: dict[str, Any]) -> str:
    """Format structured route evidence as markdown for summarization."""
    lines = [
        f"- Target SMILES: {route_input.get('target_smiles', 'unknown')}",
        f"- Contracted unique reactions: {route_input.get('contracted_unique_reactions', 'unknown')}",
        f"- Route depth: {route_input.get('route_depth', 'unknown')}",
        f"- Average template occurrence: {route_input.get('average_template_occurrence', 'unknown')}",
        "",
        "## Route Steps",
    ]
    for index, step in enumerate(route_input.get("route_steps", []), start=1):
        lines.extend(_format_step(index, step))
    lines.extend(["", "## Starting Materials"])
    for material in route_input.get("starting_materials", []):
        stock = "buyable" if material.get("in_stock") else "not marked buyable"
        lines.append(f"- {material.get('smiles', 'unknown')} ({stock})")
    return "\n".join(lines)


def _format_step(index: int, step: dict[str, Any]) -> list[str]:
    lines = [
        f"### Step {index}",
        f"- Depth: {step.get('depth', 'unknown')}",
        f"- Product: {step.get('product_smiles', 'unknown')}",
        f"- Precursors: {', '.join(step.get('precursor_smiles', [])) or 'none'}",
        f"- Template: {step.get('template') or 'unspecified'}",
        f"- Template occurrence: {step.get('template_occurrence', 'unknown')}",
        f"- Contracted reaction count: {step.get('contracted_reaction_count', 'unknown')}",
    ]
    evidence = step.get("patent_evidence", [])
    if evidence:
        lines.append("- Patent evidence:")
        for item in evidence:
            lines.extend(_format_patent_evidence(item))
    return lines


def _format_patent_evidence(evidence: dict[str, Any]) -> list[str]:
    lines = [f"  - Source: {evidence.get('source_id', 'unknown')}"]
    if evidence.get("reported_yield"):
        lines.append(f"    Reported yield: {evidence['reported_yield']}")
    lines.extend(
        _format_components("Matched precursors", evidence.get("matched_precursors", []))
    )
    lines.extend(_format_components("Products", evidence.get("product_components", [])))
    lines.extend(
        _format_components("Other components", evidence.get("other_components", []))
    )
    if evidence.get("excerpt"):
        lines.append(f"    Excerpt: {evidence['excerpt']}")
    return lines


def _format_components(
    label: str, components: list[dict[str, Any]], indent: int = 4
) -> list[str]:
    if not components:
        return []
    pad = " " * indent
    lines = [f"{pad}{label}:"]
    for component in components:
        name = component.get("name") or component.get("smiles") or "unknown"
        details = [str(component.get("role", "component"))]
        if component.get("smiles") and component.get("smiles") != name:
            details.append(str(component["smiles"]))
        quantities = _format_quantities(component.get("quantities", []))
        if quantities:
            details.append(quantities)
        lines.append(f"{pad}- {name} ({'; '.join(details)})")
    return lines


def _format_quantities(quantities: list[dict[str, Any]]) -> str:
    parts = []
    for quantity in quantities:
        text = quantity.get("text")
        if text:
            parts.append(str(text))
        elif quantity.get("type") and quantity.get("value") is not None:
            parts.append(f"{quantity['type']} {quantity['value']}")
    return ", ".join(parts)
