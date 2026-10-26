"""Rich tree view of a run: workflow nodes as branches, LLM/MCP/tool calls as leaves."""

from collections.abc import Sequence

from rich.markup import escape
from rich.tree import Tree

from buddler.models.enums import RunStatus, TraceEventType
from buddler.schemas.trace import RunOut, TraceEventOut

LEAF_ICONS: dict[TraceEventType, str] = {
    TraceEventType.LLM_CALL: "🤖",
    TraceEventType.EMBEDDING_CALL: "🔢",
    TraceEventType.TOOL_CALL: "🛠",
    TraceEventType.MCP_CALL: "🗄",
    TraceEventType.HOME_CALL: "🏠",
    TraceEventType.WEATHER_CALL: "🌦",
}
STATUS_STYLES: dict[RunStatus, str] = {
    RunStatus.SUCCEEDED: "green",
    RunStatus.FAILED: "red",
    RunStatus.RUNNING: "yellow",
    RunStatus.WAITING_REVIEW: "cyan",
    RunStatus.PENDING: "dim",
}


def _metrics(event: TraceEventOut) -> str:
    parts = []
    if event.tokens is not None:
        parts.append(f"{event.tokens} tok")
    if event.latency_ms is not None:
        parts.append(f"{event.latency_ms} ms")
    return f" [dim]({', '.join(parts)})[/dim]" if parts else ""


def run_label(run: RunOut) -> str:
    style = STATUS_STYLES[run.status]
    duration = f", {run.duration_ms} ms" if run.duration_ms is not None else ""
    return (
        f"[bold]Run {run.id}[/bold] {run.kind} [italic]{escape(run.subject)}[/italic] "
        f"[{style}]{run.status}[/{style}] "
        f"[dim](tokens {run.tokens_in}/{run.tokens_out}{duration})[/dim]"
    )


def render_run_tree(run: RunOut, events: Sequence[TraceEventOut]) -> Tree:
    root = Tree(run_label(run))
    # Latest branch per node name: nodes may repeat (agent loops).
    branches: dict[str, Tree] = {}
    for event in events:
        summary = escape(event.summary)
        name = escape(event.node)
        if event.type is TraceEventType.NODE_START:
            parent = branches.get(event.parent_node or "", root)
            branches[event.node] = parent.add(f"[yellow]▶ {name}[/yellow]")
        elif event.type is TraceEventType.NODE_END:
            branch = branches.get(event.node)
            label = f"[green]✔ {name}[/green] — {summary}{_metrics(event)}"
            if branch is None:
                root.add(label)
            else:
                branch.label = label
        elif event.type is TraceEventType.ERROR:
            # Leaf, so degraded-but-finished nodes keep their error visible; a later
            # NODE_END turns the label green again, a failed node stays red.
            branch = branches.get(event.node, root)
            branch.add(f"[red]✖ {summary}[/red]{_metrics(event)}")
            if branch is not root:
                branch.label = f"[red]✖ {name}[/red]"
        else:
            icon = LEAF_ICONS.get(event.type, "•")
            branches.get(event.node, root).add(f"{icon} {summary}{_metrics(event)}")
    return root
