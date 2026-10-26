"""Database-backed implementations of the agent ports. Each call is its own unit of work."""

import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from buddler.agents.ports import CommandRecord
from buddler.agents.state import (
    ExistingSymptom,
    KnowledgeAction,
    ProposedAction,
    Resolution,
    RetrievedSymptom,
    ReviewDecision,
)
from buddler.db import session_scope
from buddler.knowledge.conditions import ActionConditions, ActionParams
from buddler.models import Action, DeviceCommand, Symptom
from buddler.models.enums import KnowledgeStatus, Provenance
from buddler.repositories.embedding_cache_repository import EmbeddingCacheRepository
from buddler.repositories.knowledge_repository import KnowledgeRepository
from buddler.repositories.resolution_repository import (
    DeviceCommandRepository,
    ResolutionRepository,
)

SessionFactory = async_sessionmaker[AsyncSession]


def knowledge_action(action: Action) -> KnowledgeAction:
    return KnowledgeAction(
        id=action.id,
        kind=action.kind,
        title=action.title,
        instruction=action.instruction,
        params=ActionParams.model_validate(action.params),
        conditions=ActionConditions.model_validate(action.conditions),
        priority=action.priority,
        status=action.status,
    )


def action_row(
    proposal: ProposedAction,
    *,
    symptom_id: uuid.UUID,
    status: KnowledgeStatus,
    provenance: Provenance,
    prompt_version: str | None,
) -> Action:
    return Action(
        symptom_id=symptom_id,
        kind=proposal.kind,
        title=proposal.title,
        instruction=proposal.instruction,
        params=proposal.params.model_dump(mode="json", exclude_none=True),
        conditions=proposal.conditions.model_dump(mode="json", exclude_none=True),
        priority=proposal.priority,
        status=status,
        provenance=provenance,
        prompt_version=prompt_version,
        reviewed_at=None if status is KnowledgeStatus.PROPOSED else datetime.now(UTC),
    )


class DbRetrievalPort:
    def __init__(self, factory: SessionFactory) -> None:
        self._factory = factory

    async def similar_symptoms(
        self, embedding: Sequence[float], *, top_k: int, max_distance: float
    ) -> list[RetrievedSymptom]:
        async with session_scope(self._factory) as session:
            matches = await KnowledgeRepository(session).nearest_symptoms(
                embedding, top_k=top_k, max_distance=max_distance
            )
            return [
                RetrievedSymptom(
                    ref="",
                    symptom_id=m.symptom.id,
                    text=m.symptom.text,
                    distance=m.distance,
                    actions=[
                        knowledge_action(a)
                        for a in m.symptom.actions
                        if a.status is KnowledgeStatus.CONFIRMED
                    ],
                )
                for m in matches
            ]


def _symptom_status(decisions: Sequence[ReviewDecision]) -> KnowledgeStatus:
    statuses = {d.status for d in decisions}
    if KnowledgeStatus.CONFIRMED in statuses:
        return KnowledgeStatus.CONFIRMED
    if statuses == {KnowledgeStatus.REJECTED}:
        return KnowledgeStatus.REJECTED
    return KnowledgeStatus.PROPOSED


class DbTrainKnowledgePort:
    def __init__(self, factory: SessionFactory) -> None:
        self._factory = factory

    async def nearest_any(
        self, embedding: Sequence[float], max_distance: float
    ) -> ExistingSymptom | None:
        async with session_scope(self._factory) as session:
            matches = await KnowledgeRepository(session).nearest_symptoms(
                embedding, top_k=1, max_distance=max_distance, statuses=tuple(KnowledgeStatus)
            )
            if not matches:
                return None
            symptom = matches[0].symptom
            return ExistingSymptom(
                symptom_id=symptom.id,
                text=symptom.text,
                status=symptom.status,
                distance=matches[0].distance,
                actions=[knowledge_action(a) for a in symptom.actions],
            )

    async def save_proposals(
        self,
        *,
        symptom_text: str,
        embedding: Sequence[float],
        existing_id: uuid.UUID | None,
        proposals: Sequence[ProposedAction],
        decisions: Sequence[ReviewDecision],
        prompt_version: str | None,
    ) -> tuple[uuid.UUID, list[uuid.UUID]]:
        status_by_ref = {d.ref: d.status for d in decisions}
        async with session_scope(self._factory) as session:
            repo = KnowledgeRepository(session)
            symptom = (
                await repo.get_symptom(existing_id)
                if existing_id
                else await repo.get_symptom_by_text(symptom_text)
            )
            wanted = _symptom_status(decisions)
            if symptom is None:
                symptom = await repo.add_symptom(
                    Symptom(
                        text=symptom_text,
                        embedding=list(embedding),
                        status=wanted,
                        provenance=Provenance.LLM,
                        prompt_version=prompt_version,
                        reviewed_at=None
                        if wanted is KnowledgeStatus.PROPOSED
                        else datetime.now(UTC),
                    )
                )
            elif symptom.status is KnowledgeStatus.PROPOSED and wanted is KnowledgeStatus.CONFIRMED:
                # A human confirmed an action: the symptom it belongs to is confirmed with it.
                await repo.set_symptom_status([symptom.id], KnowledgeStatus.CONFIRMED)
            action_ids = await repo.add_actions(
                [
                    action_row(
                        p,
                        symptom_id=symptom.id,
                        status=status_by_ref.get(p.ref, KnowledgeStatus.PROPOSED),
                        provenance=Provenance.LLM,
                        prompt_version=prompt_version,
                    )
                    for p in proposals
                ]
            )
            return symptom.id, action_ids


class DbCommandLogPort:
    def __init__(self, factory: SessionFactory) -> None:
        self._factory = factory

    async def last_applied_at(self, room: str) -> datetime | None:
        async with session_scope(self._factory) as session:
            return await DeviceCommandRepository(session).last_applied_at(room)

    async def record(self, command: CommandRecord) -> uuid.UUID:
        async with session_scope(self._factory) as session:
            return await DeviceCommandRepository(session).add(
                DeviceCommand(
                    run_id=command.run_id,
                    action_id=command.action_id,
                    room=command.room,
                    parameter=command.parameter,
                    old_value=command.old_value,
                    new_value=command.new_value,
                    status=command.status,
                    reason=command.reason,
                )
            )


class DbResolutionPort:
    def __init__(self, factory: SessionFactory) -> None:
        self._factory = factory

    async def save(self, run_id: uuid.UUID, resolution: Resolution) -> uuid.UUID:
        citations = [
            {"ref": a.ref, "refs": a.citations, "uncited": a.uncited} for a in resolution.actions
        ] + [
            {"advice": a.text, "refs": a.citations, "uncited": a.uncited} for a in resolution.advice
        ]
        async with session_scope(self._factory) as session:
            return await ResolutionRepository(session).add(
                run_id, resolution.model_dump(mode="json"), citations
            )


class DbEmbeddingStore:
    def __init__(self, factory: SessionFactory) -> None:
        self._factory = factory

    async def get_many(self, model: str, text_hashes: Sequence[str]) -> dict[str, list[float]]:
        async with session_scope(self._factory) as session:
            return await EmbeddingCacheRepository(session).get_many(model, text_hashes)

    async def put_many(self, model: str, vectors: Mapping[str, list[float]]) -> None:
        async with session_scope(self._factory) as session:
            await EmbeddingCacheRepository(session).put_many(model, vectors)
