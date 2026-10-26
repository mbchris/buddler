"""Train workflow graph (design §3). Linear by design: easy to follow in the demonstrator."""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from buddler.agents.graphs.builder import assemble_linear, noop_node
from buddler.agents.nodes.train import TrainNodes
from buddler.agents.ports import TrainDeps
from buddler.agents.state import TrainState
from buddler.agents.tracing import traced_node

TRAIN_NODE_ORDER = ("embed", "match", "propose", "review", "persist")

TrainGraph = CompiledStateGraph[TrainState, None, TrainState, TrainState]


def build_train_graph(
    deps: TrainDeps, checkpointer: BaseCheckpointSaver[str] | None = None
) -> TrainGraph:
    """`checkpointer` is required for the interactive review interrupt."""
    nodes = TrainNodes(deps)
    return assemble_linear(
        TrainState,
        TRAIN_NODE_ORDER,
        lambda name: traced_node(name, deps.tracer, getattr(nodes, name)),
        checkpointer,
    )


def train_graph_mermaid() -> str:
    """Workflow structure only; nodes are never executed."""
    graph = assemble_linear(TrainState, TRAIN_NODE_ORDER, lambda name: noop_node)
    return graph.get_graph().draw_mermaid()
