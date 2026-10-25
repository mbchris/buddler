"""Persistence for runs and their trace events."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from buddler.models import Run, TraceEvent
from buddler.models.enums import RunKind, RunStatus
from buddler.models.run import SUBJECT_LENGTH
from buddler.tracing.events import TraceEventData


class RunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        kind: RunKind,
        subject: str,
        status: RunStatus = RunStatus.RUNNING,
    ) -> Run:
        run = Run(kind=kind, subject=subject[:SUBJECT_LENGTH], status=status)
        self._session.add(run)
        await self._session.flush()
        return run

    async def get(self, run_id: uuid.UUID) -> Run | None:
        return await self._session.get(Run, run_id)

    async def finish(
        self,
        run_id: uuid.UUID,
        status: RunStatus,
        *,
        tokens_in: int,
        tokens_out: int,
        error: str | None = None,
    ) -> None:
        await self._session.execute(
            update(Run)
            .where(Run.id == run_id)
            .values(
                status=status,
                finished_at=datetime.now(UTC),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                error=error,
            )
        )

    async def set_status(self, run_id: uuid.UUID, status: RunStatus) -> None:
        await self._session.execute(update(Run).where(Run.id == run_id).values(status=status))

    async def list_recent(self, limit: int = 20) -> list[Run]:
        query = select(Run).order_by(Run.started_at.desc()).limit(limit)
        return list((await self._session.scalars(query)).all())

    async def add_event(self, event: TraceEventData) -> None:
        self._session.add(
            TraceEvent(
                run_id=event.run_id,
                seq=event.seq,
                node=event.node,
                parent_node=event.parent_node,
                type=event.type,
                summary=event.summary,
                payload=event.payload,
                tokens=event.tokens,
                latency_ms=event.latency_ms,
                ts=event.ts,
            )
        )

    async def list_events(self, run_id: uuid.UUID, after_seq: int = 0) -> list[TraceEvent]:
        query = (
            select(TraceEvent)
            .where(TraceEvent.run_id == run_id, TraceEvent.seq > after_seq)
            .order_by(TraceEvent.seq)
        )
        return list((await self._session.scalars(query)).all())
