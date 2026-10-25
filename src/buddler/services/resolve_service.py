"""Run the resolve workflow; MCP is optional (the agent degrades to `read_room` only)."""

import uuid
from collections.abc import Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from buddler.agents.graphs.resolve import build_resolve_graph
from buddler.agents.ports import Clock, KnowledgeQueryPort, ResolveDeps
from buddler.agents.state import AgentFindings, Resolution
from buddler.config import Settings
from buddler.home.factory import open_home
from buddler.home.ports import HomePort
from buddler.llm.factory import build_chat_client, build_embedding_client
from buddler.mcp_client.client import McpUnavailableError, connect_knowledge_mcp
from buddler.models.enums import RunKind
from buddler.services.db_ports import (
    DbCommandLogPort,
    DbEmbeddingStore,
    DbResolutionPort,
    DbRetrievalPort,
    SessionFactory,
)
from buddler.services.run_service import traced_run
from buddler.tracing.tracer import Tracer, TraceSink
from buddler.weather.factory import open_weather
from buddler.weather.ports import WeatherPort


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


@dataclass(frozen=True)
class ResolveRequest:
    situation: str
    room: str | None = None
    # Simulated local time for demos ("what would happen at 02:00?").
    at: datetime | None = None
    dry_run: bool = False
    use_mcp: bool = True


@dataclass(frozen=True)
class ResolveOutcome:
    run_id: uuid.UUID
    resolution_id: uuid.UUID
    resolution: Resolution
    agent: AgentFindings
    candidates: int


@dataclass(frozen=True)
class Adapters:
    """Injected in tests; otherwise built from settings."""

    home: HomePort | None = None
    weather: WeatherPort | None = None
    clock: Clock | None = None


async def _connect_mcp(
    stack: AsyncExitStack, settings: Settings, tracer: Tracer
) -> KnowledgeQueryPort | None:
    try:
        async with tracer.span("mcp_connect", "connecting to Postgres MCP server") as span:
            client = await stack.enter_async_context(
                connect_knowledge_mcp(
                    settings.mcp_url,
                    max_rows=settings.mcp_max_rows,
                    timeout_s=settings.mcp_timeout_s,
                )
            )
            span.summary = f"connected to {settings.mcp_url} (tool: execute_sql)"
    except McpUnavailableError:
        # Already traced as the span's ERROR event; the agent prompt states the tool is off.
        return None
    return client


async def run_resolve(
    request: ResolveRequest,
    *,
    settings: Settings,
    factory: SessionFactory,
    extra_sinks: Sequence[TraceSink] = (),
    run_id: uuid.UUID | None = None,
    adapters: Adapters | None = None,
) -> ResolveOutcome:
    async with (
        traced_run(
            factory, RunKind.RESOLVE, request.situation, extra_sinks=extra_sinks, run_id=run_id
        ) as tracer,
        AsyncExitStack() as stack,
    ):
        adapters = adapters or Adapters()
        home = adapters.home
        if home is None:
            home = await stack.enter_async_context(open_home(settings))
        weather = adapters.weather
        if weather is None:
            weather = await stack.enter_async_context(open_weather(settings))
        knowledge_query = await _connect_mcp(stack, settings, tracer) if request.use_mcp else None
        deps = ResolveDeps(
            chat=build_chat_client(settings),
            embedder=build_embedding_client(settings, DbEmbeddingStore(factory)),
            home=home,
            weather=weather,
            clock=adapters.clock or SystemClock(),
            retrieval=DbRetrievalPort(factory),
            commands=DbCommandLogPort(factory),
            resolutions=DbResolutionPort(factory),
            tracer=tracer,
            settings=settings,
            knowledge_query=knowledge_query,
        )
        state: dict[str, Any] = await build_resolve_graph(deps).ainvoke(
            {
                "situation": request.situation,
                "room_argument": request.room,
                "at": request.at,
                "dry_run": request.dry_run,
            }
        )
    return ResolveOutcome(
        run_id=tracer.run_id,
        resolution_id=state["resolution_id"],
        resolution=state["resolution"],
        agent=state["agent"],
        candidates=len(state["candidates"]),
    )
