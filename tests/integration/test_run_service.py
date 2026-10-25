"""traced_run commits in its own sessions, so these tests clean up the rows they create."""

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from buddler.db import create_session_factory
from buddler.llm.ports import LLMUsage
from buddler.models import Run
from buddler.models.enums import RunKind, RunStatus, TraceEventType
from buddler.repositories.run_repository import RunRepository
from buddler.services.run_service import traced_run


@pytest.fixture
async def factory(engine: AsyncEngine) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    factory = create_session_factory(engine)
    yield factory
    async with factory() as session:
        await session.execute(delete(Run))
        await session.commit()


async def test_successful_run_persists_events_and_totals(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    async with traced_run(factory, RunKind.TRAIN, "feeling cold") as tracer, tracer.span("propose"):
        await tracer.llm_call(LLMUsage(model="m", prompt_tokens=7, completion_tokens=2), "x")

    async with factory() as session:
        repo = RunRepository(session)
        run = await repo.get(tracer.run_id)
        events = await repo.list_events(tracer.run_id)
    assert run is not None
    assert (run.status, run.tokens_in, run.tokens_out) == (RunStatus.SUCCEEDED, 7, 2)
    assert run.finished_at is not None
    assert [e.type for e in events] == [
        TraceEventType.NODE_START,
        TraceEventType.LLM_CALL,
        TraceEventType.NODE_END,
    ]


async def test_failed_run_keeps_events_and_records_error(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    with pytest.raises(RuntimeError):
        async with (
            traced_run(factory, RunKind.RESOLVE, "I am cold") as tracer,
            tracer.span("agent"),
        ):
            raise RuntimeError("model unavailable")

    async with factory() as session:
        repo = RunRepository(session)
        run = await repo.get(tracer.run_id)
        events = await repo.list_events(tracer.run_id, after_seq=1)
    assert run is not None
    assert (run.status, run.error) == (RunStatus.FAILED, "RuntimeError: model unavailable")
    assert [e.type for e in events] == [TraceEventType.ERROR]
