import uuid
from datetime import UTC, datetime

import pytest
from rich.console import Console

from buddler.llm.ports import LLMUsage
from buddler.models.enums import RunKind, RunStatus, TraceEventType
from buddler.schemas.trace import RunOut, TraceEventOut
from buddler.tracing.events import MAX_PAYLOAD_CHARS, TraceEventData, bounded_payload
from buddler.tracing.render import render_run_tree
from buddler.tracing.tracer import MemoryTraceSink, Tracer, null_tracer

RUN_ID = uuid.UUID(int=1)


def _tracer() -> tuple[Tracer, MemoryTraceSink]:
    sink = MemoryTraceSink()
    return Tracer(RUN_ID, [sink]), sink


def _shape(sink: MemoryTraceSink) -> list[tuple[int, str, str | None, TraceEventType]]:
    return [(e.seq, e.node, e.parent_node, e.type) for e in sink.events]


async def test_nested_spans_record_parents_in_order() -> None:
    tracer, sink = _tracer()

    async with tracer.span("resolve"), tracer.span("retrieve") as result:
        await tracer.emit(TraceEventType.MCP_CALL, "SELECT 1")
        result.summary = "3 matches"

    assert _shape(sink) == [
        (1, "resolve", None, TraceEventType.NODE_START),
        (2, "retrieve", "resolve", TraceEventType.NODE_START),
        (3, "retrieve", None, TraceEventType.MCP_CALL),
        (4, "retrieve", "resolve", TraceEventType.NODE_END),
        (5, "resolve", None, TraceEventType.NODE_END),
    ]
    assert sink.events[3].summary == "3 matches"
    assert sink.events[3].latency_ms is not None


async def test_span_error_is_traced_reraised_and_context_restored() -> None:
    tracer, sink = _tracer()

    with pytest.raises(ValueError, match="boom"):
        async with tracer.span("detect"):
            raise ValueError("boom")

    last = sink.events[-1]
    assert (last.type, last.summary) == (TraceEventType.ERROR, "ValueError: boom")
    assert tracer.current_node is None


async def test_llm_call_accumulates_token_totals() -> None:
    tracer, sink = _tracer()

    await tracer.llm_call(LLMUsage(model="m", prompt_tokens=10, completion_tokens=3), "a")
    await tracer.llm_call(LLMUsage(model="m", prompt_tokens=5, completion_tokens=2), "b")

    assert (tracer.totals.tokens_in, tracer.totals.tokens_out) == (15, 5)
    assert sink.events[0].tokens == 13
    assert sink.events[0].payload["model"] == "m"


class BrokenSink:
    async def write(self, event: TraceEventData) -> None:
        raise ConnectionError("db down")


async def test_failing_sink_never_breaks_workflow() -> None:
    healthy = MemoryTraceSink()
    tracer = Tracer(RUN_ID, [BrokenSink(), healthy])

    async with tracer.span("ingest"):
        pass

    assert len(healthy.events) == 2


async def test_null_tracer_accepts_all_calls() -> None:
    tracer = null_tracer()

    async with tracer.span("x"):
        await tracer.emit(TraceEventType.TOOL_CALL, "noop")


@pytest.mark.parametrize(
    ("payload", "truncated"),
    [({"a": 1}, False), ({"big": "x" * (MAX_PAYLOAD_CHARS + 1)}, True)],
)
def test_payload_is_bounded(payload: dict[str, object], truncated: bool) -> None:
    assert ("truncated" in bounded_payload(payload)) is truncated


def test_payload_values_are_made_json_safe() -> None:
    assert bounded_payload({"id": RUN_ID}) == {"id": str(RUN_ID)}


def _event(seq: int, node: str, type_: TraceEventType, parent: str | None = None) -> TraceEventOut:
    return TraceEventOut(
        seq=seq,
        node=node,
        parent_node=parent,
        type=type_,
        summary=f"{node} {type_}",
        payload={},
        tokens=None,
        latency_ms=1,
        ts=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_render_tree_shows_nodes_leaves_and_errors() -> None:
    run = RunOut(
        id=RUN_ID,
        kind=RunKind.RESOLVE,
        subject="I'm feeling cold",
        status=RunStatus.FAILED,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        finished_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
        tokens_in=1,
        tokens_out=2,
        error="boom",
    )
    events = [
        _event(1, "embed", TraceEventType.NODE_START),
        _event(2, "embed", TraceEventType.EMBEDDING_CALL),
        _event(3, "embed", TraceEventType.NODE_END),
        _event(4, "agent", TraceEventType.NODE_START),
        _event(5, "agent", TraceEventType.ERROR),
    ]
    console = Console(record=True, width=120)

    console.print(render_run_tree(run, events))

    text = console.export_text()
    assert "failed" in text and "1000 ms" in text and "I'm feeling cold" in text
    assert "✔ embed" in text and "🔢 embed embedding_call" in text
    assert "✖ agent" in text and "✖ agent error" in text
