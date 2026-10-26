"""Typer CLI entry point."""

import asyncio
import math
import uuid

import typer
import uvicorn
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Column, Table
from rich.text import Text

from buddler import __version__
from buddler.agents.graphs.resolve import resolve_graph_mermaid
from buddler.agents.graphs.train import train_graph_mermaid
from buddler.cli.ask import ask
from buddler.cli.home import home_app, weather
from buddler.cli.knowledge import knowledge_app
from buddler.cli.train import train
from buddler.config import get_settings
from buddler.db import (
    EmbeddingDimensionMismatchError,
    create_engine,
    create_session_factory,
    ensure_embedding_dim,
    session_scope,
    upgrade_database,
)
from buddler.llm.factory import build_chat_client, build_embedding_client
from buddler.llm.ports import ChatMessage, LLMError
from buddler.repositories.run_repository import RunRepository
from buddler.schemas.trace import RunOut, TraceEventOut
from buddler.tracing.render import STATUS_STYLES, render_run_tree

app = typer.Typer(
    name="buddler", no_args_is_help=True, help="Agentic Homematic assistant demonstrator."
)
db_app = typer.Typer(no_args_is_help=True, help="Database schema management.")
app.add_typer(db_app, name="db")
llm_app = typer.Typer(no_args_is_help=True, help="LLM gateway utilities.")
app.add_typer(llm_app, name="llm")
runs_app = typer.Typer(no_args_is_help=True, help="Inspect workflow runs and their traces.")
app.add_typer(runs_app, name="runs")
app.add_typer(knowledge_app, name="knowledge")
app.add_typer(home_app, name="home")
app.command()(ask)
app.command()(train)
app.command()(weather)
graph_app = typer.Typer(no_args_is_help=True, help="Print workflow graphs as Mermaid.")
app.add_typer(graph_app, name="graph")
console = Console()


@app.command()
def version() -> None:
    """Print the application version."""
    console.print(__version__)


@app.command("config")
def show_config() -> None:
    """Show effective settings (secrets masked)."""
    settings = get_settings()
    table = Table("Setting", "Value")
    for name, value in settings.model_dump().items():
        table.add_row(name, str(value))
    console.print(table)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Start the REST API and trace UI."""
    uvicorn.run("buddler.api.main:app", factory=True, host=host, port=port, reload=reload)


@db_app.command("upgrade")
def db_upgrade(revision: str = typer.Argument("head")) -> None:
    """Apply Alembic migrations (tables, pgvector indexes, knowledge views, grants)."""
    settings = get_settings()
    try:
        ensure_embedding_dim(settings)
    except EmbeddingDimensionMismatchError as err:
        console.print(f"[red]{err}[/red]")
        raise typer.Exit(code=1) from err
    upgrade_database(str(settings.database_url), revision)
    console.print(f"[green]Database upgraded to {revision}.[/green]")


SMOKE_TEXTS = (
    "feeling cold",
    "I'm freezing in here",
    "the air is stuffy",
)


class SmokeVerdict(BaseModel):
    symptom: str = Field(description="Short English phrase for the person's state.")
    wants_warmer: bool


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return dot / norm


async def _smoke() -> Table:
    settings = get_settings()
    embedder = build_embedding_client(settings)
    chat = build_chat_client(settings)
    embedded = await embedder.embed(list(SMOKE_TEXTS))
    verdict = await chat.complete_structured(
        [
            ChatMessage(role="system", content="Summarize what the person feels."),
            ChatMessage(role="user", content=SMOKE_TEXTS[1]),
        ],
        SmokeVerdict,
        max_tokens=100,
    )
    vectors = embedded.vectors
    table = Table("Check", "Result", title=f"LLM smoke test ({settings.llm_provider})")
    table.add_row("embedding model", embedder.model)
    table.add_row("dimension", str(len(vectors[0])))
    if embedded.usage:
        table.add_row(
            "embedding tokens / latency",
            f"{embedded.usage.prompt_tokens} / {embedded.usage.latency_ms} ms",
        )
    table.add_row("cos(feeling cold, freezing)", f"{_cosine(vectors[0], vectors[1]):.3f}")
    table.add_row("cos(feeling cold, stuffy air)", f"{_cosine(vectors[0], vectors[2]):.3f}")
    table.add_row("chat model", verdict.usage.model)
    table.add_row("structured output", verdict.value.model_dump_json())
    table.add_row(
        "chat tokens in/out / latency / attempts",
        f"{verdict.usage.prompt_tokens}/{verdict.usage.completion_tokens}"
        f" / {verdict.usage.latency_ms} ms / {verdict.attempts}",
    )
    return table


@llm_app.command("smoke")
def llm_smoke() -> None:
    """One embedding batch + one structured chat call against the configured gateway."""
    try:
        console.print(asyncio.run(_smoke()))
    except LLMError as err:
        console.print(f"[red]{err}[/red]")
        raise typer.Exit(code=1) from err


async def _load_runs(limit: int) -> list[RunOut]:
    engine = create_engine(get_settings())
    try:
        async with session_scope(create_session_factory(engine)) as session:
            runs = await RunRepository(session).list_recent(limit)
            return [RunOut.model_validate(run) for run in runs]
    finally:
        await engine.dispose()


async def _load_run(run_id: uuid.UUID) -> tuple[RunOut, list[TraceEventOut]] | None:
    engine = create_engine(get_settings())
    try:
        async with session_scope(create_session_factory(engine)) as session:
            repo = RunRepository(session)
            run = await repo.get(run_id)
            if run is None:
                return None
            events = await repo.list_events(run_id)
            return (
                RunOut.model_validate(run),
                [TraceEventOut.model_validate(e) for e in events],
            )
    finally:
        await engine.dispose()


@runs_app.command("list")
def runs_list(limit: int = typer.Option(20, min=1)) -> None:
    """Most recent runs."""
    table = Table("kind", "subject", "status", "started", "duration", "tokens in/out")
    # Full id is needed for `runs show`; never truncate it.
    table.columns.insert(0, Column("run_id", no_wrap=True, min_width=36))
    for run in asyncio.run(_load_runs(limit)):
        style = STATUS_STYLES[run.status]
        duration = f"{run.duration_ms} ms" if run.duration_ms is not None else "-"
        table.add_row(
            str(run.id),
            run.kind,
            run.subject,
            f"[{style}]{run.status}[/{style}]",
            run.started_at.strftime("%Y-%m-%d %H:%M:%S"),
            duration,
            f"{run.tokens_in}/{run.tokens_out}",
        )
    console.print(table)


@runs_app.command("show")
def runs_show(run_id: uuid.UUID) -> None:
    """Tree of a run's trace events."""
    loaded = asyncio.run(_load_run(run_id))
    if loaded is None:
        console.print(f"[red]Run {run_id} not found.[/red]")
        raise typer.Exit(code=1)
    run, events = loaded
    console.print(render_run_tree(run, events))
    if run.error:
        console.print(Text(run.error, style="red"))


@graph_app.command("train")
def graph_train() -> None:
    """Mermaid source of the train workflow."""
    console.print(train_graph_mermaid(), markup=False, highlight=False)


@graph_app.command("resolve")
def graph_resolve() -> None:
    """Mermaid source of the resolve workflow."""
    console.print(resolve_graph_mermaid(), markup=False, highlight=False)
