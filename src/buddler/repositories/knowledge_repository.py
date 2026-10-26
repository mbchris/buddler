"""Persistence and similarity search for curated symptoms and actions."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from buddler.models import Action, Symptom
from buddler.models.enums import KnowledgeStatus


@dataclass(frozen=True)
class SymptomMatch:
    symptom: Symptom
    distance: float


class KnowledgeRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def nearest_symptoms(
        self,
        embedding: Sequence[float],
        *,
        top_k: int,
        max_distance: float,
        statuses: Sequence[KnowledgeStatus] = (KnowledgeStatus.CONFIRMED,),
    ) -> list[SymptomMatch]:
        """Closest symptoms by cosine distance, with all their actions loaded."""
        distance = Symptom.embedding.cosine_distance(list(embedding))
        query = (
            select(Symptom, distance.label("distance"))
            .options(selectinload(Symptom.actions))
            .where(Symptom.status.in_(statuses), Symptom.embedding.is_not(None))
            .where(distance <= max_distance)
            .order_by(distance)
            .limit(top_k)
        )
        rows = await self._session.execute(query)
        return [SymptomMatch(symptom=row[0], distance=float(row[1])) for row in rows]

    async def get_symptom(self, symptom_id: uuid.UUID) -> Symptom | None:
        query = (
            select(Symptom).options(selectinload(Symptom.actions)).where(Symptom.id == symptom_id)
        )
        return await self._session.scalar(query)

    async def get_symptom_by_text(self, text: str) -> Symptom | None:
        query = select(Symptom).options(selectinload(Symptom.actions)).where(Symptom.text == text)
        return await self._session.scalar(query)

    async def get_action(self, action_id: uuid.UUID) -> Action | None:
        return await self._session.get(Action, action_id)

    async def list_symptoms(
        self, *, status: KnowledgeStatus | None = None, limit: int | None = 100
    ) -> list[Symptom]:
        """Newest first; `status` filters symptoms that have it or an action with it."""
        query = (
            select(Symptom)
            .options(selectinload(Symptom.actions))
            .order_by(Symptom.created_at.desc(), Symptom.id)
        )
        if status is not None:
            with_action = select(Action.symptom_id).where(Action.status == status)
            query = query.where((Symptom.status == status) | Symptom.id.in_(with_action))
        if limit is not None:
            query = query.limit(limit)
        return list((await self._session.scalars(query)).all())

    async def add_symptom(self, symptom: Symptom) -> Symptom:
        self._session.add(symptom)
        await self._session.flush()
        return symptom

    async def add_actions(self, actions: Sequence[Action]) -> list[uuid.UUID]:
        self._session.add_all(actions)
        await self._session.flush()
        return [action.id for action in actions]

    async def set_symptom_status(
        self, symptom_ids: Sequence[uuid.UUID], status: KnowledgeStatus
    ) -> int:
        result = await self._session.execute(
            update(Symptom)
            .where(Symptom.id.in_(symptom_ids))
            .values(status=status, reviewed_at=datetime.now(UTC))
            .returning(Symptom.id)
        )
        return len(result.all())

    async def set_action_status(
        self, action_ids: Sequence[uuid.UUID], status: KnowledgeStatus
    ) -> int:
        result = await self._session.execute(
            update(Action)
            .where(Action.id.in_(action_ids))
            .values(status=status, reviewed_at=datetime.now(UTC))
            .returning(Action.id)
        )
        return len(result.all())
