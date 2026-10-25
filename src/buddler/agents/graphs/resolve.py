"""Resolve workflow graph (design §3)."""

from langgraph.graph.state import CompiledStateGraph

from buddler.agents.graphs.builder import assemble_linear, noop_node
from buddler.agents.nodes.resolve import ResolveNodes
from buddler.agents.ports import ResolveDeps
from buddler.agents.state import ResolveState
from buddler.agents.tracing import traced_node

RESOLVE_NODE_ORDER = (
    "understand",
    "locate",
    "observe",
    "retrieve",
    "evaluate",
    "agent",
    "decide",
    "act",
    "report",
)

ResolveGraph = CompiledStateGraph[ResolveState, None, ResolveState, ResolveState]


def build_resolve_graph(deps: ResolveDeps) -> ResolveGraph:
    nodes = ResolveNodes(deps)
    return assemble_linear(
        ResolveState,
        RESOLVE_NODE_ORDER,
        lambda name: traced_node(name, deps.tracer, getattr(nodes, name)),
    )


def resolve_graph_mermaid() -> str:
    """Workflow structure only; nodes are never executed."""
    graph = assemble_linear(ResolveState, RESOLVE_NODE_ORDER, lambda name: noop_node)
    return graph.get_graph().draw_mermaid()
