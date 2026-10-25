"""REST request/response schemas (separate from ORM models)."""

import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from buddler.agents.state import Resolution, ReviewMode
from buddler.models.enums import RunKind, RunStatus
from buddler.schemas.trace import RunOut

MAX_SITUATION_CHARS = 500


class ResolveRunCreate(BaseModel):
    kind: Literal[RunKind.RESOLVE]
    situation: str = Field(min_length=1, max_length=MAX_SITUATION_CHARS)
    room: str | None = None
    # Simulated time for demos; naive values are read in the configured TIMEZONE.
    at: datetime | None = None
    dry_run: bool = False
    use_mcp: bool = True


class TrainRunCreate(BaseModel):
    kind: Literal[RunKind.TRAIN]
    symptom: str = Field(min_length=1, max_length=255)
    hint: str | None = Field(default=None, max_length=MAX_SITUATION_CHARS)
    # Interactive review needs a console; review via PATCH /knowledge/... instead.
    review_mode: Literal[ReviewMode.AUTO_CONFIRM, ReviewMode.DEFER] = ReviewMode.DEFER


RunCreate = Annotated[ResolveRunCreate | TrainRunCreate, Field(discriminator="kind")]


class RunAccepted(BaseModel):
    run_id: uuid.UUID
    status: RunStatus
    status_url: str
    stream_url: str


class RunDetail(BaseModel):
    run: RunOut
    resolution: Resolution | None = None


class GraphStepOut(BaseModel):
    node: str
    title: str
    summary: str
    description: str
    why: str
    example: str
    details: list[str]


class GlossaryTermOut(BaseModel):
    term: str
    definition: str


class GraphOut(BaseModel):
    kind: RunKind
    mermaid: str
    overview: str | None = None
    glossary: list[GlossaryTermOut] = []
    steps: list[GraphStepOut] = []
