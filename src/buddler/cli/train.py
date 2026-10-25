"""`buddler train SYMPTOM`: propose actions for a symptom, with human review in the console."""

import asyncio
import sys

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from buddler.agents.state import ProposedAction, ReviewDecision, ReviewMode
from buddler.cli._db import cli_session_factory
from buddler.cli.knowledge import STATUS_STYLES, describe_conditions, styled
from buddler.config import get_settings
from buddler.llm.ports import LLMError
from buddler.models.enums import KnowledgeStatus
from buddler.services.train_service import TrainOutcome, TrainRequest, run_training

console = Console()

BULK_CHOICES = {"a": "confirm all", "e": "review each", "d": "defer all", "x": "reject all"}
EACH_CHOICES = {
    "c": KnowledgeStatus.CONFIRMED,
    "r": KnowledgeStatus.REJECTED,
    "s": KnowledgeStatus.PROPOSED,
}


def _params(proposal: ProposedAction) -> str:
    return ", ".join(f"{k}={v:g}" for k, v in proposal.params.model_dump(exclude_none=True).items())


def _proposal_table(proposals: list[ProposedAction]) -> Table:
    table = Table("ref", "kind", "prio", "title", "conditions", title="Proposed actions")
    for p in proposals:
        params = _params(p)
        table.add_row(
            str(p.ref),
            p.kind + (f" ({params})" if params else ""),
            str(p.priority),
            p.title,
            describe_conditions(p.conditions.model_dump(mode="json", exclude_none=True)),
        )
    return table


def _ask_decisions(proposals: list[ProposedAction]) -> list[ReviewDecision]:
    console.print(_proposal_table(proposals))
    choice = Prompt.ask(
        " / ".join(f"[{key}] {label}" for key, label in BULK_CHOICES.items()),
        choices=list(BULK_CHOICES),
        default="e",
    )
    bulk = {
        "a": KnowledgeStatus.CONFIRMED,
        "d": KnowledgeStatus.PROPOSED,
        "x": KnowledgeStatus.REJECTED,
    }
    if choice in bulk:
        return [ReviewDecision(ref=p.ref, status=bulk[choice]) for p in proposals]
    decisions = []
    for p in proposals:
        console.print(f"\n[bold]#{p.ref} {p.title}[/bold] ({p.kind})")
        console.print(f"  {p.instruction}", markup=False, highlight=False)
        answer = Prompt.ask(
            "  [c]onfirm / [r]eject / [s]kip", choices=list(EACH_CHOICES), default="c"
        )
        decisions.append(ReviewDecision(ref=p.ref, status=EACH_CHOICES[answer]))
    return decisions


async def _console_reviewer(proposals: list[ProposedAction]) -> list[ReviewDecision]:
    # Prompts block on stdin; keep the event loop free.
    return await asyncio.to_thread(_ask_decisions, proposals)


async def _train(request: TrainRequest, mode: ReviewMode) -> TrainOutcome:
    async with cli_session_factory() as factory:
        return await run_training(
            request, mode, settings=get_settings(), factory=factory, reviewer=_console_reviewer
        )


def train(
    symptom: str = typer.Argument(..., help="Symptom phrase, e.g. 'feeling cold'."),
    hint: str | None = typer.Option(None, help="Guidance for the proposals."),
    review: ReviewMode | None = typer.Option(
        None, help="interactive (default on a TTY), auto = confirm all, defer = keep proposed."
    ),
) -> None:
    """Let the AI propose actions for SYMPTOM and curate them into the knowledge base."""
    mode = review or (ReviewMode.INTERACTIVE if sys.stdin.isatty() else ReviewMode.DEFER)
    try:
        outcome = asyncio.run(_train(TrainRequest(symptom=symptom, hint=hint), mode))
    except LLMError as err:
        console.print(f"[red]{err}[/red]")
        raise typer.Exit(code=1) from err
    counts = {s: sum(d.status is s for d in outcome.decisions) for s in KnowledgeStatus}
    target = "existing" if outcome.existing else "new"
    console.print(
        f"\n[bold]Run {outcome.run_id}[/bold]: {target} symptom '{outcome.symptom_text}', "
        f"{len(outcome.proposals)} proposals -> "
        + (
            ", ".join(styled(f"{n} {s}", STATUS_STYLES[s]) for s, n in counts.items() if n)
            or "none"
        )
    )
    console.print(f"Trace: buddler runs show {outcome.run_id}", markup=False)
