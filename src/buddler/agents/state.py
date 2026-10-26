"""Workflow state and the value objects it carries.

Pydantic models so the LangGraph checkpointer can serialize them. State holds small
summaries; the one vector (train: the symptom embedding) is never traced.
"""

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Literal, TypedDict

from pydantic import BaseModel, Field

from buddler.knowledge.conditions import (
    ActionConditions,
    ActionParams,
    ConditionCheck,
    Observations,
)
from buddler.models.enums import ActionKind, CommandStatus, KnowledgeStatus

Urgency = Literal["low", "normal", "high"]


class ReviewMode(StrEnum):
    INTERACTIVE = "interactive"
    AUTO_CONFIRM = "auto"
    # Persist as `proposed`; review later via `knowledge confirm/reject` or API.
    DEFER = "defer"


class KnowledgeAction(BaseModel):
    id: uuid.UUID
    kind: ActionKind
    title: str
    instruction: str
    params: ActionParams
    conditions: ActionConditions
    priority: int
    status: KnowledgeStatus = KnowledgeStatus.CONFIRMED


class ToolCallRecord(BaseModel):
    step: int
    tool: str
    argument: str
    is_error: bool
    row_count: int | None
    latency_ms: int


class AgentFindings(BaseModel):
    notes: str
    steps: int
    tool_calls: list[ToolCallRecord]
    stopped_reason: str


# --- resolve -----------------------------------------------------------------


class Understanding(BaseModel):
    symptom: str
    room_mention: str | None
    urgency: Urgency
    # False when the LLM failed and the raw sentence is used as symptom.
    from_llm: bool = True


class RoomSource(StrEnum):
    ARGUMENT = "argument"
    MENTIONED = "mentioned"
    DEFAULT = "default"
    UNKNOWN = "unknown"


class RetrievedSymptom(BaseModel):
    """A matched symptom as shown to the LLM; `ref` (S1, S2, …) is what citations point to."""

    ref: str
    symptom_id: uuid.UUID
    text: str
    distance: float
    actions: list[KnowledgeAction]


class Candidate(BaseModel):
    """A knowledge action evaluated against this run's observations (ref A1, A2, …)."""

    ref: str
    symptom_ref: str
    action: KnowledgeAction
    checks: list[ConditionCheck]
    # Advice: no condition violated. Device action: guard passed (all met, within bounds).
    applicable: bool
    reason: str
    target_c: float | None = None


class ActionOutcome(StrEnum):
    ADVISED = "advised"
    APPLIED = CommandStatus.APPLIED.value
    DRY_RUN = CommandStatus.DRY_RUN.value
    REJECTED = CommandStatus.REJECTED.value
    FAILED = CommandStatus.FAILED.value


class ChosenAction(BaseModel):
    ref: str
    action_id: uuid.UUID | None
    kind: ActionKind | None
    title: str
    rationale: str
    citations: list[str]
    # AGENTS.md §6: claims without a valid citation are flagged, not hidden.
    uncited: bool
    # Filled by `act`; `rejected` until then for refs that are not applicable.
    outcome: ActionOutcome | None = None
    detail: str = ""


class AdviceItem(BaseModel):
    text: str
    citations: list[str]
    uncited: bool


class Resolution(BaseModel):
    symptom: str
    room: str | None
    reply: str
    summary: str
    actions: list[ChosenAction]
    advice: list[AdviceItem]
    observations: Observations
    confidence: float = Field(ge=0, le=1)
    prompt_versions: dict[str, str]
    uncited_count: int
    invalid_refs: list[str] = Field(default_factory=list)
    dry_run: bool = False


class ResolveState(TypedDict, total=False):
    situation: str
    room_argument: str | None
    at: datetime | None
    dry_run: bool
    understanding: Understanding
    room: str | None
    room_source: RoomSource
    observations: Observations
    retrieved: list[RetrievedSymptom]
    candidates: list[Candidate]
    agent: AgentFindings
    resolution: Resolution
    resolution_id: uuid.UUID


# --- train -------------------------------------------------------------------


class ExistingSymptom(BaseModel):
    symptom_id: uuid.UUID
    text: str
    status: KnowledgeStatus
    distance: float
    actions: list[KnowledgeAction]


class ProposedAction(BaseModel):
    ref: int
    kind: ActionKind
    title: str
    instruction: str
    params: ActionParams
    conditions: ActionConditions
    priority: int


class ReviewDecision(BaseModel):
    ref: int
    status: KnowledgeStatus


class TrainState(TypedDict, total=False):
    symptom: str
    hint: str | None
    review_mode: ReviewMode
    embedding: list[float]
    existing: ExistingSymptom | None
    # Normalized phrase proposed by the LLM (stored text for a new symptom).
    normalized_symptom: str
    proposals: list[ProposedAction]
    prompt_version: str
    decisions: list[ReviewDecision]
    saved_symptom_id: uuid.UUID
    saved_action_ids: list[uuid.UUID]
