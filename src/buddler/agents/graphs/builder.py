"""Linear LangGraph assembly shared by all workflows."""

from collections.abc import Callable, Sequence
from itertools import pairwise
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.typing import StateLike  # type: ignore[attr-defined]  # public module, no __all__

from buddler.agents.tracing import NodeUpdate, StateNode


def assemble_linear[S: StateLike](
    state_schema: type[S],
    order: Sequence[str],
    node_for: Callable[[str], StateNode[S]],
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[S, None, S, S]:
    graph: StateGraph[S, None, S, S] = StateGraph(state_schema)
    for name in order:
        graph.add_node(name, node_for(name))
    graph.add_edge(START, order[0])
    for current, following in pairwise(order):
        graph.add_edge(current, following)
    graph.add_edge(order[-1], END)
    return graph.compile(checkpointer=checkpointer)


async def noop_node(state: Any) -> NodeUpdate:
    """Placeholder for structure-only graphs (Mermaid export)."""
    return {}
