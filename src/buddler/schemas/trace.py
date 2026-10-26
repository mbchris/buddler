"""I/O schemas for runs and trace events (CLI renderer, REST API, trace UI)."""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from buddler.models.enums import RunKind, RunStatus, TraceEventType


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    kind: RunKind
    subject: str
    status: RunStatus
    started_at: datetime
    finished_at: datetime | None
    tokens_in: int
    tokens_out: int
    error: str | None

    @property
    def duration_ms(self) -> int | None:
        if self.finished_at is None:
            return None
        return int((self.finished_at - self.started_at).total_seconds() * 1000)


class TraceEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    seq: int
    node: str
    parent_node: str | None
    type: TraceEventType
    summary: str
    # Schemaless by design: content differs per event type.
    payload: dict[str, Any]
    tokens: int | None
    latency_ms: int | None
    ts: datetime
