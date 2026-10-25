"""Wrap LangGraph nodes in tracer spans (AGENTS.md §8)."""

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from langgraph.errors import GraphBubbleUp

from buddler.tracing.tracer import SpanResult, Tracer

# Partial state update returned by a LangGraph node.
NodeUpdate = dict[str, Any]


class StateNode[S](Protocol):
    """LangGraph calls nodes with a parameter named `state` (Callable[[S], ...] won't match)."""

    def __call__(self, state: S) -> Awaitable[NodeUpdate]: ...


PAUSED_SUMMARY = "paused: waiting for human input"


def traced_node[S](
    name: str, tracer: Tracer, fn: Callable[[S, SpanResult], Awaitable[NodeUpdate]]
) -> StateNode[S]:
    """Run `fn(state, span)` inside a span.

    LangGraph's `interrupt()` raises GraphBubbleUp to pause the graph. That is control
    flow, not a failure: the span closes as "paused" and the signal is re-raised after it.
    On resume LangGraph re-runs the node, which opens a second span for it.
    """

    async def run(state: S) -> NodeUpdate:
        paused: GraphBubbleUp | None = None
        async with tracer.span(name) as span:
            try:
                return await fn(state, span)
            except GraphBubbleUp as signal:
                span.summary = PAUSED_SUMMARY
                paused = signal
        # Spans never suppress exceptions; only the pause path reaches this point.
        assert paused is not None
        raise paused

    run.__name__ = name
    return run
