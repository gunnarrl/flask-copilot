import asyncio
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable, TYPE_CHECKING

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
    def __init__(
        self,
        experiment: "FlaskExperiment",
        callback: "AgentCallbackType" = None,
    ):
        self.experiment = experiment
        self.callback = callback

    async def summarize(self, candidate: RouteCandidate) -> RouteSummary:
        prompt = build_summary_prompt(route_summary_input(candidate))
        try:
            from charge.tasks.task import Task

            task = Task(
                system_prompt="You summarize retrosynthesis routes concisely and only use the provided route data.",
                user_prompt=prompt,
            )
            if self.callback is None:
                agent = self.experiment.create_agent_with_experiment_state(task=task)
            else:
                agent = self.experiment.create_agent_with_experiment_state(
                    task=task,
                    callback=self.callback,
                )
            result = await agent.run()
        except Exception as exc:
            return RouteSummary(
                route_id=candidate.route_id,
                summary=None,
                error=str(exc),
            )
        return RouteSummary(route_id=candidate.route_id, summary=str(result).strip())


async def summarize_routes(
    candidates: list[RouteCandidate],
    experiment: "FlaskExperiment",
    limit: int = 10,
    concurrency: int = 4,
    callback: "AgentCallbackType" = None,
    status_callback: Callable[[str], Awaitable[None]] | None = None,
) -> list[SummarizedRoute]:
    semaphore = asyncio.Semaphore(concurrency)
    summarizer = RouteSummarizer(experiment, callback)
    selected_candidates = candidates[:limit]
    total = len(selected_candidates)

    async def summarize_one(
        index: int, candidate: RouteCandidate
    ) -> SummarizedRoute:
        async with semaphore:
            if status_callback is not None:
                await status_callback(
                    f"Summarizer {index} started summarizing Route {index} of {total}."
                )
            summary = await summarizer.summarize(candidate)
            if callback is not None:
                call_id = f"summarize_route:{candidate.route_id}"
                if summary.error:
                    await callback.on_tool_result(
                        "summarize_route",
                        summary.error,
                        is_error=True,
                        source=f"Summarizer {index}",
                        call_id=call_id,
                    )
                else:
                    await callback.on_tool_call(
                        "summarize_route",
                        {"route_number": index, "route_id": candidate.route_id},
                        source=f"Summarizer {index}",
                        call_id=call_id,
                    )
                    await callback.on_tool_result(
                        "summarize_route",
                        summary.summary,
                        source=f"Summarizer {index}",
                        call_id=call_id,
                    )
            return SummarizedRoute(candidate=candidate, summary=summary)

    return await asyncio.gather(
        *(
            summarize_one(index, candidate)
            for index, candidate in enumerate(selected_candidates, start=1)
        )
    )


def route_summary_input(candidate: RouteCandidate) -> dict[str, Any]:
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
    lines = [
        f"  - Source: {evidence.get('source_id', 'unknown')}",
        f"    Quality rank: {evidence.get('quality_rank', 'unknown')}",
    ]
    if evidence.get("reported_yield"):
        lines.append(f"    Reported yield: {evidence['reported_yield']}")
    lines.extend(
        _format_components("Matched precursors", evidence.get("matched_precursors", []))
    )
    lines.extend(_format_components("Products", evidence.get("product_components", [])))
    lines.extend(
        _format_components("Other components", evidence.get("other_components", []))
    )
    for action in evidence.get("actions", []):
        text = action.get("text") or ""
        lines.append(f"    Action: {action.get('type', 'unspecified')} - {text}")
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
