"""Knowledge curation: list, review, and import human-authored entries (AGENTS.md §4)."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from buddler.agents.state import ProposedAction
from buddler.llm.ports import EmbeddingClient
from buddler.models import Symptom
from buddler.models.enums import KnowledgeStatus, Provenance
from buddler.repositories.knowledge_repository import KnowledgeRepository
from buddler.schemas.knowledge import SeedFile, SymptomOut
from buddler.services.db_ports import action_row


class KnowledgeNotFoundError(LookupError):
    pass


@dataclass(frozen=True)
class ImportSummary:
    created: list[str]
    # Existing symptoms are never modified by an import.
    skipped: list[str]
    actions: int


@dataclass(frozen=True)
class ReviewSummary:
    symptoms: int
    actions: int
    missing: list[uuid.UUID]


class KnowledgeService:
    def __init__(self, session: AsyncSession) -> None:
        self._repo = KnowledgeRepository(session)

    async def list_symptoms(
        self, status: KnowledgeStatus | None = None, limit: int | None = 100
    ) -> list[SymptomOut]:
        return [
            SymptomOut.model_validate(s)
            for s in await self._repo.list_symptoms(status=status, limit=limit)
        ]

    async def review_symptom(self, symptom_id: uuid.UUID, status: KnowledgeStatus) -> SymptomOut:
        if not await self._repo.set_symptom_status([symptom_id], status):
            raise KnowledgeNotFoundError(f"Symptom {symptom_id} not found.")
        symptom = await self._repo.get_symptom(symptom_id)
        assert symptom is not None
        return SymptomOut.model_validate(symptom)

    async def review_action(self, action_id: uuid.UUID, status: KnowledgeStatus) -> SymptomOut:
        action = await self._repo.get_action(action_id)
        if action is None:
            raise KnowledgeNotFoundError(f"Action {action_id} not found.")
        await self._repo.set_action_status([action_id], status)
        symptom = await self._repo.get_symptom(action.symptom_id)
        assert symptom is not None
        if status is KnowledgeStatus.CONFIRMED and symptom.status is KnowledgeStatus.PROPOSED:
            # Confirming an action of a proposed symptom confirms the symptom too.
            await self._repo.set_symptom_status([symptom.id], KnowledgeStatus.CONFIRMED)
        refreshed = await self._repo.get_symptom(action.symptom_id)
        assert refreshed is not None
        return SymptomOut.model_validate(refreshed)

    async def review(self, ids: Sequence[uuid.UUID], status: KnowledgeStatus) -> ReviewSummary:
        """CLI convenience: each id may be a symptom or an action."""
        symptoms = actions = 0
        missing: list[uuid.UUID] = []
        for item_id in ids:
            try:
                await self.review_action(item_id, status)
                actions += 1
                continue
            except KnowledgeNotFoundError:
                pass
            try:
                await self.review_symptom(item_id, status)
                symptoms += 1
            except KnowledgeNotFoundError:
                missing.append(item_id)
        return ReviewSummary(symptoms=symptoms, actions=actions, missing=missing)

    async def import_seed(self, seed: SeedFile, embedder: EmbeddingClient) -> ImportSummary:
        """Human-curated entries are stored as confirmed with provenance `human`."""
        new: list[str] = []
        skipped: list[str] = []
        for entry in seed.symptoms:
            if await self._repo.get_symptom_by_text(entry.text) is None:
                new.append(entry.text)
            else:
                skipped.append(entry.text)
        by_text = {s.text: s for s in seed.symptoms}
        vectors = (await embedder.embed(new)).vectors if new else []
        now = datetime.now(UTC)
        actions = 0
        for text, vector in zip(new, vectors, strict=True):
            symptom = await self._repo.add_symptom(
                Symptom(
                    text=text,
                    embedding=vector,
                    status=KnowledgeStatus.CONFIRMED,
                    provenance=Provenance.HUMAN,
                    reviewed_at=now,
                )
            )
            rows = [
                action_row(
                    ProposedAction(ref=index, **action.model_dump()),
                    symptom_id=symptom.id,
                    status=KnowledgeStatus.CONFIRMED,
                    provenance=Provenance.HUMAN,
                    prompt_version=None,
                )
                for index, action in enumerate(by_text[text].actions, start=1)
            ]
            actions += len(await self._repo.add_actions(rows))
        return ImportSummary(created=new, skipped=skipped, actions=actions)
