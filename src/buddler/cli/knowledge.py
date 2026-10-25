"""`buddler knowledge …`: list, import and review symptom -> action knowledge."""

import asyncio
import json
import uuid
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from buddler.cli._db import cli_session_factory
from buddler.config import get_settings
from buddler.db import session_scope
from buddler.llm.factory import build_embedding_client
from buddler.models.enums import KnowledgeStatus
from buddler.schemas.knowledge import ActionOut, SeedFile, SymptomOut
from buddler.services.db_ports import DbEmbeddingStore
from buddler.services.knowledge_service import ImportSummary, KnowledgeService, ReviewSummary

console = Console()
knowledge_app = typer.Typer(no_args_is_help=True, help="Curate symptom -> action knowledge.")

STATUS_STYLES = {
    KnowledgeStatus.CONFIRMED: "green",
    KnowledgeStatus.PROPOSED: "yellow",
    KnowledgeStatus.REJECTED: "red",
}


def styled(text: str, style: str) -> str:
    return f"[{style}]{text}[/{style}]" if style else text


def describe_params(action: ActionOut) -> str:
    params = action.params.model_dump(exclude_none=True)
    return ", ".join(f"{k}={v:+g}" if k == "delta_c" else f"{k}={v:g}" for k, v in params.items())


def describe_conditions(conditions: dict[str, object]) -> str:
    return ", ".join(f"{k}={v}" for k, v in conditions.items()) or "always"


def symptom_table(symptoms: list[SymptomOut]) -> Table:
    table = Table("symptom / action", "kind", "prio", "conditions", "status", "id")
    for symptom in symptoms:
        table.add_row(
            f"[bold]{symptom.text}[/bold]",
            "",
            "",
            "",
            styled(symptom.status, STATUS_STYLES[symptom.status]),
            str(symptom.id),
        )
        for action in symptom.actions:
            params = describe_params(action)
            table.add_row(
                f"  {action.title}",
                action.kind + (f" ({params})" if params else ""),
                str(action.priority),
                describe_conditions(action.conditions.model_dump(mode="json", exclude_none=True)),
                styled(action.status, STATUS_STYLES[action.status]),
                str(action.id),
            )
    return table


async def _list(status: KnowledgeStatus | None, limit: int) -> list[SymptomOut]:
    async with cli_session_factory() as factory, session_scope(factory) as session:
        return await KnowledgeService(session).list_symptoms(status, limit)


@knowledge_app.command("list")
def list_knowledge(
    status: KnowledgeStatus | None = typer.Option(None, help="Filter symptoms/actions."),
    limit: int = typer.Option(100, min=1),
) -> None:
    """Symptoms with their actions, newest first."""
    symptoms = asyncio.run(_list(status, limit))
    if not symptoms:
        console.print(
            "Knowledge base is empty. Try: buddler knowledge import data/knowledge/seed.json"
        )
        return
    console.print(symptom_table(symptoms))


async def _import(seed: SeedFile) -> ImportSummary:
    settings = get_settings()
    async with cli_session_factory() as factory:
        embedder = build_embedding_client(settings, DbEmbeddingStore(factory))
        async with session_scope(factory) as session:
            return await KnowledgeService(session).import_seed(seed, embedder)


@knowledge_app.command("import")
def import_knowledge(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
) -> None:
    """Import human-curated knowledge (JSON) as confirmed; existing symptoms are skipped."""
    try:
        seed = SeedFile.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValidationError) as err:
        console.print(f"[red]Invalid knowledge file: {err}[/red]")
        raise typer.Exit(code=1) from err
    summary = asyncio.run(_import(seed))
    console.print(
        f"Imported {len(summary.created)} symptoms with {summary.actions} actions; "
        f"skipped {len(summary.skipped)} existing"
        + (f" ({', '.join(summary.skipped)})" if summary.skipped else "")
        + "."
    )


async def _review(ids: list[uuid.UUID], status: KnowledgeStatus) -> ReviewSummary:
    async with cli_session_factory() as factory, session_scope(factory) as session:
        return await KnowledgeService(session).review(ids, status)


def _print_review(summary: ReviewSummary, status: KnowledgeStatus) -> None:
    console.print(
        f"{styled(status, STATUS_STYLES[status])}: "
        f"{summary.actions} actions, {summary.symptoms} symptoms."
    )
    if summary.missing:
        console.print(f"[red]Not found: {', '.join(map(str, summary.missing))}[/red]")
        raise typer.Exit(code=1)


@knowledge_app.command("confirm")
def confirm(ids: list[uuid.UUID] = typer.Argument(..., help="Symptom or action ids.")) -> None:
    """Confirm entries (only confirmed knowledge is used by `ask`)."""
    _print_review(asyncio.run(_review(ids, KnowledgeStatus.CONFIRMED)), KnowledgeStatus.CONFIRMED)


@knowledge_app.command("reject")
def reject(ids: list[uuid.UUID] = typer.Argument(..., help="Symptom or action ids.")) -> None:
    """Reject entries."""
    _print_review(asyncio.run(_review(ids, KnowledgeStatus.REJECTED)), KnowledgeStatus.REJECTED)
