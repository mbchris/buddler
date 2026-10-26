"""Run tracer: ordered events, nested spans, token totals, fan-out to sinks.

Tracing must never change workflow behavior: sink failures are logged and swallowed,
and `null_tracer()` has no sinks at all.
"""

import asyncio
import contextvars
import logging
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from buddler.llm.ports import LLMUsage
from buddler.models.enums import TraceEventType
from buddler.tracing.events import JsonPayload, TraceEventData, bounded_payload

logger = logging.getLogger(__name__)

_current_node: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "trace_current_node", default=None
)


class TraceSink(Protocol):
    async def write(self, event: TraceEventData) -> None: ...


@dataclass
class TokenTotals:
    tokens_in: int = 0
    tokens_out: int = 0


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


class Tracer:
    def __init__(self, run_id: uuid.UUID, sinks: Sequence[TraceSink]) -> None:
        self.run_id = run_id
        self._sinks = tuple(sinks)
        self._seq = 0
        self._lock = asyncio.Lock()
        self.totals = TokenTotals()

    @property
    def current_node(self) -> str | None:
        return _current_node.get()

    async def emit(
        self,
        type_: TraceEventType,
        summary: str,
        *,
        node: str | None = None,
        payload: JsonPayload | None = None,
        tokens: int | None = None,
        latency_ms: int | None = None,
    ) -> None:
        own_node = node or _current_node.get() or "run"
        parent = _current_node.get() if node else None
        async with self._lock:
            self._seq += 1
            event = TraceEventData(
                run_id=self.run_id,
                seq=self._seq,
                node=own_node,
                parent_node=parent if parent != own_node else None,
                type=type_,
                summary=summary,
                ts=datetime.now(UTC),
                payload=bounded_payload(payload or {}),
                tokens=tokens,
                latency_ms=latency_ms,
            )
            for sink in self._sinks:
                try:
                    await sink.write(event)
                except Exception:
                    logger.exception("Trace sink %s failed; event %d dropped", sink, event.seq)

    @asynccontextmanager
    async def span(
        self, node: str, summary: str = "", payload: JsonPayload | None = None
    ) -> AsyncIterator["SpanResult"]:
        """NODE_START/NODE_END (or ERROR) around a workflow step; nested events get it as parent."""
        await self.emit(
            TraceEventType.NODE_START, summary or f"{node} started", node=node, payload=payload
        )
        token = _current_node.set(node)
        started = time.perf_counter()
        result = SpanResult()
        try:
            try:
                yield result
            finally:
                _current_node.reset(token)
        except Exception as err:
            await self.emit(
                TraceEventType.ERROR,
                f"{type(err).__name__}: {err}",
                node=node,
                payload={"error_type": type(err).__name__},
                latency_ms=_elapsed_ms(started),
            )
            raise
        await self.emit(
            TraceEventType.NODE_END,
            result.summary or f"{node} finished",
            node=node,
            payload=result.payload,
            latency_ms=_elapsed_ms(started),
        )

    async def llm_call(
        self,
        usage: LLMUsage,
        summary: str,
        *,
        type_: TraceEventType = TraceEventType.LLM_CALL,
        payload: JsonPayload | None = None,
    ) -> None:
        self.totals.tokens_in += usage.prompt_tokens
        self.totals.tokens_out += usage.completion_tokens
        await self.emit(
            type_,
            summary,
            payload={
                "model": usage.model,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                **(payload or {}),
            },
            tokens=usage.prompt_tokens + usage.completion_tokens,
            latency_ms=usage.latency_ms,
        )


@dataclass
class SpanResult:
    """Filled by the span body; becomes the NODE_END summary/payload."""

    summary: str = ""
    payload: JsonPayload | None = None


def null_tracer() -> Tracer:
    """Tracing disabled: same API, no sinks."""
    return Tracer(uuid.UUID(int=0), sinks=())


class MemoryTraceSink:
    def __init__(self) -> None:
        self.events: list[TraceEventData] = []

    async def write(self, event: TraceEventData) -> None:
        self.events.append(event)
