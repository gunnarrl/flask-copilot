import argparse
import asyncio
import base64
import mimetypes
from pathlib import Path

from charge_backend.attachments import validate_image_attachments
from charge.clients.agentframework import AgentFrameworkBackend
from lc_conductor import ToolRuntime, resolve_builtin_tool_descriptors
from lc_conductor.resolve_default_parameters import resolve_orchestrator_config

from charge_backend.builtin_tools import list_builtin_tool_definitions
from charge_backend.flask_experiment import FlaskExperiment
from charge_backend.retrosynthesis.ai import retrosynthesis


def attachment_from_path(path: str) -> dict:
    file_path = Path(path)
    data = file_path.read_bytes()
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(data).decode("ascii")

    return {
        "id": file_path.stem,
        "name": file_path.name,
        "mimeType": mime_type,
        "sizeBytes": len(data),
        "dataUrl": f"data:{mime_type};base64,{encoded}",
        "createdAt": "",
    }


async def run_retrosynthesis_command(args: argparse.Namespace) -> None:
    excluded_tools = args.excluded_tools
    builtin_tool_definitions = [
        tool
        for tool in list_builtin_tool_definitions()
        if tool.identifier not in excluded_tools
    ]
    tools = ToolRuntime(
        tools=(
            []
            if args.no_tools
            else resolve_builtin_tool_descriptors(None, builtin_tool_definitions)
        )
    )

    attachment_paths = args.attachments or [] # attachments can be none
    attachments = validate_image_attachments({
        "attachments": [
            attachment_from_path(path) for path in attachment_paths
        ]
    })

    # copied from 'lc_conductor/lc_conductor/backend_manager.py'
    cfg = resolve_orchestrator_config(
        requested_backend=args.backend,
        requested_model=args.model,
        return_api_key=True,
    )

    agent_backend = AgentFrameworkBackend(
        model=cfg["model"],
        backend=cfg["backend"],
        api_key=cfg["apiKey"],
        base_url=cfg["baseUrl"],
        use_responses_api=True,
    )

    experiment = FlaskExperiment(task=None, backend=agent_backend)
    runner = experiment.create_agent_with_experiment_state(
        task=None,
        agent_key="cli:retrosynthesis",
    )

    res, _, _ = await retrosynthesis(
        target=args.target,
        query=args.query,
        constraint=args.constraint,
        attachments=attachments,
        runner=runner,
        tool_runtime=tools,
    )

    if args.json:
        print(res.model_dump_json(indent=2))
    else:
        print("Reasoning:", res.reasoning_summary)
        print("Reactants:", ", ".join(res.reactants_smiles_list))
        print("Products:", ", ".join(res.products_smiles_list))


def main() -> None:
    parser = argparse.ArgumentParser(prog="flask-copilot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    retro = subparsers.add_parser("retrosynthesis")
    retro.add_argument(
        "--model",
        type=str,
        default="gpt-5.4",
        help="Model to use for the orchestrator",
    )
    retro.add_argument(
        "--backend",
        type=str,
        default="livai",
        help="Backend to use for the orchestrator client",
    )
    retro.add_argument("target", type=str, help="SMILES string for target molecule")

    retro.add_argument("--query", type=str, help="Additional user requirements")

    retro.add_argument(
        "--no-tools", action="store_true", help="Disable all tools"
    )

    retro.add_argument(
        "--excluded-tools",
        nargs="*",
        default=[],
        help="List of tools to disable"
    )

    retro.add_argument(
        "--constraint",
        type=str,
        help="Additional user constraints"
    )

    retro.add_argument(
        "--attachments",
        type=str,
        nargs="*",
        help="List of filepaths to attachments (image only)"
    )

    retro.add_argument(
        "--json",
        action="store_true",
        help="Print the json dump directly"
    )

    args = parser.parse_args()
    if args.command == "retrosynthesis":
        asyncio.run(run_retrosynthesis_command(args))


if __name__ == "__main__":
    main()
