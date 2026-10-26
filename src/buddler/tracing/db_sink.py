"""Persist trace events in their own short transactions.

Independent of the workflow's session: events survive a failed/rolled-back step
and are visible to live readers (CLI, SSE) immediately.
"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from buddler.db import session_scope
from buddler.repositories.run_repository import RunRepository
from buddler.tracing.events import TraceEventData


class DbTraceSink:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def write(self, event: TraceEventData) -> None:
        async with session_scope(self._session_factory) as session:
            await RunRepository(session).add_event(event)
