from flask_tools.chemistry.smiles_utils import (
    canonicalize_smiles,
    verify_smiles,
)
from typing import Awaitable, Callable, TYPE_CHECKING

from charge_backend.moleculedb.purchasable import is_purchasable
from charge_backend.retrosynthesis import route_planner
from charge_backend.retrosynthesis.database import query_reaction_database
from charge_backend.retrosynthesis.template import enumerate_template_route_candidates
from charge_backend.pdf import PdfDocumentRegistry
from lc_conductor import (
    BuiltinToolDefinition,
    doc_summary,
)

if TYPE_CHECKING:
    from charge_backend.flask_experiment import FlaskExperiment


def _document_consult_tool(registry: PdfDocumentRegistry):
    async def consult_with_document(
        question: str,
        document_id: str | None = None,
        max_tool_calls: int = 14,
    ) -> str:
        """Consult the active uploaded PDF reference with a private subagent."""
        return await registry.consult(question, document_id, max_tool_calls)

    return consult_with_document


def _enumerate_template_routes_tool(
    config_file: str,
    experiment: "FlaskExperiment",
    status_callback: Callable[[str], Awaitable[None]] | None = None,
):
    async def enumerate_template_routes(
        smiles: str,
        k: int = 10,
    ) -> str:
        """Enumerate and summarize ranked template routes for a target SMILES."""
        if status_callback is not None:
            await status_callback(f"Enumerating {k} template routes for {smiles}.")
        candidates = await enumerate_template_route_candidates(config_file, smiles, k)
        if not candidates:
            return f"No template routes found for {smiles}."
        planner_entry = experiment.agent_registry.get("route-planning:planner")
        callback = planner_entry.agent.callback if planner_entry is not None else None
        summarized_routes = await route_planner.summarize_routes(
            candidates,
            experiment,
            limit=k,
            concurrency=min(k, 10),
            callback=callback,
        )
        return route_planner.build_route_context(summarized_routes, limit=k).format()

    return enumerate_template_routes


def list_builtin_tool_definitions(
    pdf_registry: PdfDocumentRegistry | None = None,
    retrosynthesis_config_file: str | None = None,
    experiment: "FlaskExperiment | None" = None,
    route_planning_status_callback: Callable[[str], Awaitable[None]] | None = None,
) -> list[BuiltinToolDefinition]:
    definitions = [
        BuiltinToolDefinition(
            identifier="verify_smiles",
            function=verify_smiles,
            label="Verify SMILES",
            description=doc_summary(verify_smiles),
        ),
        BuiltinToolDefinition(
            identifier="canonicalize_smiles",
            function=canonicalize_smiles,
            label="Canonicalize SMILES",
            description=doc_summary(canonicalize_smiles),
        ),
        BuiltinToolDefinition(
            identifier="is_purchasable",
            function=is_purchasable,
            label="Check Purchasability",
            description=doc_summary(is_purchasable),
        ),
        BuiltinToolDefinition(
            identifier="query_reaction_database",
            function=query_reaction_database,
            label="Query Reaction Database",
            description=doc_summary(query_reaction_database),
        ),
    ]
    if retrosynthesis_config_file is not None and experiment is not None:
        enumerate_template_routes = _enumerate_template_routes_tool(
            retrosynthesis_config_file,
            experiment,
            route_planning_status_callback,
        )
        definitions.append(
            BuiltinToolDefinition(
                identifier="enumerate_template_routes",
                function=enumerate_template_routes,
                label="Enumerate Template Routes",
                description=doc_summary(enumerate_template_routes),
            )
        )
    if pdf_registry is not None:
        consult_with_document = _document_consult_tool(pdf_registry)
        definitions.append(
            BuiltinToolDefinition(
                identifier="consult_with_document",
                function=consult_with_document,
                label="Consult Document",
                description=doc_summary(consult_with_document),
            )
        )
    return definitions
