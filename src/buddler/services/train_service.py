"""Run the train workflow, including the human review pause/resume loop."""

import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt
from pydantic import TypeAdapter

from buddler.agents.graphs.train import build_train_graph
from buddler.agents.ports import TrainDeps
from buddler.agents.state import ExistingSymptom, ProposedAction, ReviewDecision, ReviewMode
from buddler.config import Settings
from buddler.db import session_scope
from buddler.llm.factory import build_chat_client, build_embedding_client
from buddler.models.enums import RunKind, RunStatus
from buddler.repositories.run_repository import RunRepository
from buddler.services.db_ports import DbEmbeddingStore, DbTrainKnowledgePort, SessionFactory
from buddler.services.run_service import traced_run
from buddler.tracing.tracer import TraceSink

Reviewer = Callable[[list[ProposedAction]], Awaitable[list[ReviewDecision]]]

_proposals_adapter = TypeAdapter(list[ProposedAction])
INTERRUPT_KEY = "__interrupt__"


@dataclass(frozen=True)
class TrainRequest:
    symptom: str
    hint: str | None = None


@dataclass(frozen=True)
class TrainOutcome:
    run_id: uuid.UUID
    symptom_id: uuid.UUID
    symptom_text: str
    existing: ExistingSymptom | None
    proposals: list[ProposedAction]
    decisions: list[ReviewDecision]
    saved_action_ids: list[uuid.UUID]


async def _set_run_status(factory: SessionFactory, run_id: uuid.UUID, status: RunStatus) -> None:
    async with session_scope(factory) as session:
        await RunRepository(session).set_status(run_id, status)


def _pending_proposals(state: dict[str, Any]) -> list[ProposedAction]:
    interrupts: Sequence[Interrupt] = state[INTERRUPT_KEY]
    return _proposals_adapter.validate_python(interrupts[0].value["proposals"])


async def run_training(
    request: TrainRequest,
    review_mode: ReviewMode,
    *,
    settings: Settings,
    factory: SessionFactory,
    reviewer: Reviewer | None = None,
    extra_sinks: Sequence[TraceSink] = (),
    run_id: uuid.UUID | None = None,
) -> TrainOutcome:
    if review_mode is ReviewMode.INTERACTIVE and reviewer is None:
        raise ValueError("Interactive review needs a reviewer callback.")
    async with traced_run(
        factory, RunKind.TRAIN, request.symptom, extra_sinks=extra_sinks, run_id=run_id
    ) as tracer:
        deps = TrainDeps(
            chat=build_chat_client(settings),
            embedder=build_embedding_client(settings, DbEmbeddingStore(factory)),
            knowledge=DbTrainKnowledgePort(factory),
            tracer=tracer,
            settings=settings,
        )
        # In-memory checkpoints suffice: pause and resume happen in this process.
        graph = build_train_graph(deps, InMemorySaver())
        config: RunnableConfig = {"configurable": {"thread_id": str(tracer.run_id)}}
        state: dict[str, Any] = await graph.ainvoke(
            {"symptom": request.symptom, "hint": request.hint, "review_mode": review_mode},
            config,
        )
        while INTERRUPT_KEY in state:
            assert reviewer is not None  # guarded above; only interactive mode interrupts
            await _set_run_status(factory, tracer.run_id, RunStatus.WAITING_REVIEW)
            decisions = await reviewer(_pending_proposals(state))
            await _set_run_status(factory, tracer.run_id, RunStatus.RUNNING)
            state = await graph.ainvoke(
                Command(resume=[d.model_dump(mode="json") for d in decisions]), config
            )
    existing: ExistingSymptom | None = state.get("existing")
    return TrainOutcome(
        run_id=tracer.run_id,
        symptom_id=state["saved_symptom_id"],
        symptom_text=existing.text if existing else state["normalized_symptom"],
        existing=existing,
        proposals=state["proposals"],
        decisions=state["decisions"],
        saved_action_ids=state["saved_action_ids"],
    )
