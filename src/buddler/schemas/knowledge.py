"""Knowledge I/O schemas: API responses and the human-curated seed file format."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from buddler.knowledge.conditions import ActionConditions, ActionParams, validate_params
from buddler.models.enums import ActionKind, KnowledgeStatus, Provenance


class ActionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    kind: ActionKind
    title: str
    instruction: str
    params: ActionParams
    conditions: ActionConditions
    priority: int
    status: KnowledgeStatus
    provenance: Provenance
    prompt_version: str | None
    created_at: datetime
    reviewed_at: datetime | None


class SymptomOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    text: str
    status: KnowledgeStatus
    provenance: Provenance
    created_at: datetime
    reviewed_at: datetime | None
    actions: list[ActionOut]


class KnowledgeReview(BaseModel):
    status: KnowledgeStatus = Field(description="confirmed, rejected, or proposed (reset).")


class SeedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ActionKind
    title: str = Field(max_length=255)
    instruction: str
    params: ActionParams = ActionParams()
    conditions: ActionConditions = ActionConditions()
    priority: int = Field(default=100, ge=1, le=1000)

    @model_validator(mode="after")
    def _params_fit_kind(self) -> "SeedAction":
        validate_params(self.kind, self.params)
        return self


class SeedSymptom(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=255)
    actions: list[SeedAction] = Field(min_length=1)


class SeedFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symptoms: list[SeedSymptom]
