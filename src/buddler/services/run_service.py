"""Run lifecycle: create, trace, finish (succeeded/failed) with token totals."""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from buddler.db import session_scope
from buddler.models.enums import RunKind, RunStatus
from buddler.repositories.run_repository import RunRepository
from buddler.tracing.db_sink import DbTraceSink
from buddler.tracing.tracer import Tracer, TraceSink

logger = logging.getLogger(__name__)

MAX_ERROR_CHARS = 2000


@asynccontextmanager
async def traced_run(
    session_factory: async_sessionmaker[AsyncSession],
    kind: RunKind,
    subject: str,
    extra_sinks: Sequence[TraceSink] = (),
    run_id: uuid.UUID | None = None,
) -> AsyncIterator[Tracer]:
    """Yields a tracer bound to a RUNNING run; the run is finished on exit.

    `run_id` adopts a run created earlier as PENDING (API: id returned before work starts).
    """
    async with session_scope(session_factory) as session:
        repo = RunRepository(session)
        if run_id is None:
            run_id = (await repo.create(kind, subject)).id
        else:
            await repo.set_status(run_id, RunStatus.RUNNING)
    tracer = Tracer(run_id, [DbTraceSink(session_factory), *extra_sinks])
    try:
        yield tracer
    except asyncio.CancelledError:
        # Shutdown or client cancel: never leave the run RUNNING forever.
        await _finish(session_factory, tracer, RunStatus.FAILED, "cancelled")
        raise
    except Exception as err:
        await _finish(session_factory, tracer, RunStatus.FAILED, f"{type(err).__name__}: {err}")
        raise
    await _finish(session_factory, tracer, RunStatus.SUCCEEDED)


async def create_pending_run(
    session_factory: async_sessionmaker[AsyncSession],
    kind: RunKind,
    subject: str,
) -> uuid.UUID:
    async with session_scope(session_factory) as session:
        run = await RunRepository(session).create(kind, subject, RunStatus.PENDING)
        return run.id


async def _finish(
    session_factory: async_sessionmaker[AsyncSession],
    tracer: Tracer,
    status: RunStatus,
    error: str | None = None,
) -> None:
    async with session_scope(session_factory) as session:
        await RunRepository(session).finish(
            tracer.run_id,
            status,
            tokens_in=tracer.totals.tokens_in,
            tokens_out=tracer.totals.tokens_out,
            error=error[:MAX_ERROR_CHARS] if error else None,
        )
    logger.info("Run %s finished: %s", tracer.run_id, status)
