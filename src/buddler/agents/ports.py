"""Ports the workflows depend on; implemented by `services/db_ports.py`, adapters and fakes."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from buddler.agents.state import (
    ExistingSymptom,
    ProposedAction,
    Resolution,
    RetrievedSymptom,
    ReviewDecision,
)
from buddler.config import Settings
from buddler.home.ports import HomePort
from buddler.llm.ports import ChatClient, EmbeddingClient
from buddler.mcp_client.client import QueryResult
from buddler.models.enums import CommandStatus
from buddler.tracing.tracer import Tracer
from buddler.weather.ports import WeatherPort


class Clock(Protocol):
    def now(self) -> datetime:
        """Timezone-aware current time."""
        ...


class RetrievalPort(Protocol):
    async def similar_symptoms(
        self, embedding: Sequence[float], *, top_k: int, max_distance: float
    ) -> list[RetrievedSymptom]:
        """Confirmed symptoms with their confirmed actions (vector search in the repository)."""
        ...


class KnowledgeQueryPort(Protocol):
    """Read-only SQL on the `knowledge` schema via the Postgres MCP server."""

    async def execute_sql(self, sql: str) -> QueryResult: ...


@dataclass(frozen=True)
class CommandRecord:
    run_id: uuid.UUID
    action_id: uuid.UUID | None
    room: str | None
    parameter: str
    old_value: float | None
    new_value: float | None
    status: CommandStatus
    reason: str


class CommandLogPort(Protocol):
    async def last_applied_at(self, room: str) -> datetime | None: ...

    async def record(self, command: CommandRecord) -> uuid.UUID: ...


class ResolutionPort(Protocol):
    async def save(self, run_id: uuid.UUID, resolution: Resolution) -> uuid.UUID: ...


@dataclass(frozen=True)
class ResolveDeps:
    chat: ChatClient
    embedder: EmbeddingClient
    home: HomePort
    weather: WeatherPort
    clock: Clock
    retrieval: RetrievalPort
    commands: CommandLogPort
    resolutions: ResolutionPort
    tracer: Tracer
    settings: Settings
    # None when the MCP server is unreachable: the agent runs without the query tool (traced).
    knowledge_query: KnowledgeQueryPort | None


class TrainKnowledgePort(Protocol):
    async def nearest_any(
        self, embedding: Sequence[float], max_distance: float
    ) -> ExistingSymptom | None:
        """Closest symptom of any status (duplicates, context for proposals)."""
        ...

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
        """Returns (symptom id, new action ids). Never modifies existing entries' content."""
        ...


@dataclass(frozen=True)
class TrainDeps:
    chat: ChatClient
    embedder: EmbeddingClient
    knowledge: TrainKnowledgePort
    tracer: Tracer
    settings: Settings
