"""Trace event value objects (AGENTS.md §8). Payloads carry summaries, never raw input content."""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from buddler.models.enums import TraceEventType

# Keeps trace rows small and the UI responsive; oversize payloads are replaced, flagged.
MAX_PAYLOAD_CHARS = 8000

JsonPayload = dict[str, Any]


def bounded_payload(payload: JsonPayload) -> JsonPayload:
    """JSON-safe copy of `payload`; non-serializable values become strings."""
    encoded = json.dumps(payload, default=str)
    if len(encoded) > MAX_PAYLOAD_CHARS:
        return {"truncated": True, "original_chars": len(encoded), "keys": sorted(payload)}
    loaded: JsonPayload = json.loads(encoded)
    return loaded


@dataclass(frozen=True)
class TraceEventData:
    run_id: uuid.UUID
    seq: int
    node: str
    parent_node: str | None
    type: TraceEventType
    summary: str
    ts: datetime
    payload: JsonPayload = field(default_factory=dict)
    tokens: int | None = None
    latency_ms: int | None = None
