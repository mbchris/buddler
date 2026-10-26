"""Read side of runs: detail, events, and a follower for live streaming.

The follower polls the database, so runs started from the CLI in another process
stream just like API-started ones.
"""

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from buddler.agents.state import Resolution
from buddler.db import session_scope
from buddler.models.enums import RunStatus
from buddler.repositories.resolution_repository import ResolutionRepository
from buddler.repositories.run_repository import RunRepository
from buddler.schemas.trace import RunOut, TraceEventOut

FINISHED_STATUSES = frozenset({RunStatus.SUCCEEDED, RunStatus.FAILED})


class RunNotFoundError(LookupError):
    pass


@dataclass(frozen=True)
class Heartbeat:
    pass


@dataclass(frozen=True)
class StreamEnd:
    run: RunOut


StreamItem = TraceEventOut | Heartbeat | StreamEnd


class RunQueryService:
    def __init__(self, session: AsyncSession) -> None:
        self._runs = RunRepository(session)
        self._resolutions = ResolutionRepository(session)

    async def list_runs(self, limit: int) -> list[RunOut]:
        return [RunOut.model_validate(run) for run in await self._runs.list_recent(limit)]

    async def get_run(self, run_id: uuid.UUID) -> RunOut:
        run = await self._runs.get(run_id)
        if run is None:
            raise RunNotFoundError(f"Run {run_id} not found.")
        return RunOut.model_validate(run)

    async def get_resolution(self, run_id: uuid.UUID) -> Resolution | None:
        stored = await self._resolutions.get_for_run(run_id)
        return Resolution.model_validate(stored.body) if stored else None

    async def events(self, run_id: uuid.UUID, after_seq: int = 0) -> list[TraceEventOut]:
        await self.get_run(run_id)
        return [
            TraceEventOut.model_validate(event)
            for event in await self._runs.list_events(run_id, after_seq)
        ]


async def follow_run(
    factory: async_sessionmaker[AsyncSession],
    run_id: uuid.UUID,
    *,
    after_seq: int = 0,
    poll_interval_s: float,
    heartbeat_s: float,
) -> AsyncIterator[StreamItem]:
    """Yield new events until the run is finished and drained, then StreamEnd."""
    last_seq = after_seq
    last_output = time.monotonic()
    while True:
        async with session_scope(factory) as session:
            service = RunQueryService(session)
            run = await service.get_run(run_id)
            events = await service.events(run_id, last_seq)
        for event in events:
            last_seq = event.seq
            yield event
        if events:
            last_output = time.monotonic()
        elif run.status in FINISHED_STATUSES:
            # Status is read before events: an empty poll after "finished" means drained.
            yield StreamEnd(run=run)
            return
        elif time.monotonic() - last_output >= heartbeat_s:
            last_output = time.monotonic()
            yield Heartbeat()
        await asyncio.sleep(poll_interval_s)
