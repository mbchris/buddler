"""`buddler ask "I'm feeling cold"`: run the resolve workflow and print the resolution."""

import asyncio
from datetime import datetime, time

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from buddler.agents.state import ActionOutcome
from buddler.cli._db import cli_session_factory
from buddler.config import get_settings
from buddler.llm.ports import LLMError
from buddler.services.resolve_service import ResolveOutcome, ResolveRequest, run_resolve

console = Console()

OUTCOME_STYLES = {
    ActionOutcome.APPLIED: "bold green",
    ActionOutcome.DRY_RUN: "cyan",
    ActionOutcome.ADVISED: "green",
    ActionOutcome.REJECTED: "yellow",
    ActionOutcome.FAILED: "red",
}
UNCITED_MARK = " [red](uncited)[/red]"


def parse_at(value: str | None) -> datetime | None:
    """`HH:MM` = today at that local time; anything else must be ISO 8601."""
    if not value:
        return None
    settings = get_settings()
    try:
        clock = time.fromisoformat(value)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as err:
            raise typer.BadParameter("use HH:MM or an ISO datetime") from err
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=settings.tz)
    return datetime.combine(datetime.now(settings.tz).date(), clock, tzinfo=settings.tz)


async def _ask(request: ResolveRequest) -> ResolveOutcome:
    async with cli_session_factory() as factory:
        return await run_resolve(request, settings=get_settings(), factory=factory)


def _print(outcome: ResolveOutcome) -> None:
    res = outcome.resolution
    obs = res.observations
    facts = [f"{obs.local_time:%a %H:%M}" + (" (simulated)" if obs.simulated_time else "")]
    if obs.weather:
        facts.append(
            f"outside {obs.weather.outdoor_temp_c} C, sun {obs.weather.sun_hours_forecast} h"
        )
    if obs.climate:
        facts.append(
            f"{obs.climate.room}: {obs.climate.actual_c} C, setpoint {obs.climate.setpoint_c} C"
        )
    console.print(f"[dim]{' · '.join(facts)}[/dim]")
    title = f"Buddler (confidence {res.confidence:.2f})" + (" — dry run" if res.dry_run else "")
    console.print(Panel(res.reply, title=title, expand=False))
    if res.actions:
        table = Table("ref", "action", "outcome", "citations", title="Actions")
        for a in res.actions:
            style = OUTCOME_STYLES.get(a.outcome, "") if a.outcome else ""
            table.add_row(
                a.ref,
                f"[bold]{a.title}[/bold]\n{a.rationale}",
                f"[{style}]{a.outcome}[/{style}]\n{a.detail}" if style else str(a.outcome),
                ", ".join(a.citations) + (UNCITED_MARK if a.uncited else ""),
            )
        console.print(table)
    for item in res.advice:
        console.print(
            f"• {item.text} [dim]{', '.join(item.citations)}[/dim]"
            + (UNCITED_MARK if item.uncited else "")
        )
    agent = outcome.agent
    console.print(
        f"[dim]{res.summary} — symptom '{res.symptom}', room {res.room or 'unknown'}, "
        f"{outcome.candidates} candidates, agent {agent.steps} steps / "
        f"{len(agent.tool_calls)} tool calls ({agent.stopped_reason}).[/dim]"
    )
    console.print(f"Trace: buddler runs show {outcome.run_id}", markup=False)


def ask(
    situation: str = typer.Argument(..., help='What the person says, e.g. "I\'m feeling cold".'),
    room: str | None = typer.Option(None, "--room", "-r", help="Room the person is in."),
    at: str | None = typer.Option(None, help="Simulate local time: HH:MM or ISO datetime."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Log device actions, do not write."),
    mcp: bool = typer.Option(True, help="Let the agent query the knowledge DB via MCP."),
    as_json: bool = typer.Option(False, "--json", help="Print the resolution as JSON."),
) -> None:
    """Resolve a situation: observe, match knowledge, decide, act."""
    request = ResolveRequest(
        situation=situation, room=room, at=parse_at(at), dry_run=dry_run, use_mcp=mcp
    )
    try:
        outcome = asyncio.run(_ask(request))
    except LLMError as err:
        console.print(f"[red]{err}[/red]")
        raise typer.Exit(code=1) from err
    if as_json:
        console.print_json(outcome.resolution.model_dump_json())
    else:
        _print(outcome)
