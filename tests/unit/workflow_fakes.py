"""In-memory port fakes shared by the workflow tests."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, time

from buddler.agents.ports import CommandRecord, ResolveDeps, TrainDeps
from buddler.agents.state import (
    ExistingSymptom,
    KnowledgeAction,
    ProposedAction,
    Resolution,
    RetrievedSymptom,
    ReviewDecision,
)
from buddler.config import Settings
from buddler.home.fake import FakeHome
from buddler.knowledge.conditions import ActionConditions, ActionParams
from buddler.llm.fakes import FakeChatClient, FakeEmbeddingClient
from buddler.mcp_client.client import QueryResult
from buddler.models.enums import ActionKind
from buddler.tracing.tracer import MemoryTraceSink, Tracer
from buddler.weather.fake import FakeWeather

RUN_ID = uuid.UUID(int=1)
COLD_ID = uuid.UUID(int=10)
HEAT_UP = KnowledgeAction(
    id=uuid.UUID(int=11),
    kind=ActionKind.HEATING_DELTA,
    title="Raise heating by 1 C",
    instruction="Raise the setpoint by 1 C.",
    params=ActionParams(delta_c=1.0),
    conditions=ActionConditions(
        time_from=time(6), time_to=time(22), sun_hours_max=1.0, window_open=False
    ),
    priority=10,
)
TEA = KnowledgeAction(
    id=uuid.UUID(int=12),
    kind=ActionKind.ADVICE,
    title="Warm drink or sweater",
    instruction="Have a hot tea.",
    params=ActionParams(),
    conditions=ActionConditions(time_from=time(22), time_to=time(6)),
    priority=20,
)
SUN = KnowledgeAction(
    id=uuid.UUID(int=13),
    kind=ActionKind.ADVICE,
    title="Let the sun in",
    instruction="Open the blinds.",
    params=ActionParams(),
    conditions=ActionConditions(time_from=time(6), time_to=time(20), sun_hours_min=2.0),
    priority=30,
)
COLD = RetrievedSymptom(
    ref="", symptom_id=COLD_ID, text="feeling cold", distance=0.05, actions=[HEAT_UP, TEA, SUN]
)


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "llm_provider": "fake",
        "homematic_provider": "fake",
        "weather_provider": "fake",
        "default_room": "living room",
        "actuation_rooms": ["living room"],
        "timezone": "Europe/Berlin",
        "agent_max_steps": 3,
        "agent_max_tool_calls": 2,
        **overrides,
    }
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


@dataclass
class FixedClock:
    value: datetime = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)

    def now(self) -> datetime:
        return self.value


@dataclass
class FakeRetrieval:
    symptoms: list[RetrievedSymptom] = field(default_factory=lambda: [COLD])
    calls: list[tuple[int, float]] = field(default_factory=list)

    async def similar_symptoms(
        self, embedding: Sequence[float], *, top_k: int, max_distance: float
    ) -> list[RetrievedSymptom]:
        self.calls.append((top_k, max_distance))
        return [s.model_copy() for s in self.symptoms]


@dataclass
class FakeCommandLog:
    last_applied: datetime | None = None
    records: list[CommandRecord] = field(default_factory=list)

    async def last_applied_at(self, room: str) -> datetime | None:
        return self.last_applied

    async def record(self, command: CommandRecord) -> uuid.UUID:
        self.records.append(command)
        return uuid.uuid4()


@dataclass
class FakeResolutions:
    saved: list[Resolution] = field(default_factory=list)

    async def save(self, run_id: uuid.UUID, resolution: Resolution) -> uuid.UUID:
        self.saved.append(resolution)
        return uuid.UUID(int=99)


@dataclass
class FakeKnowledgeQuery:
    text: str = "[{'room': 'living room', 'status': 'applied'}]"
    sql: list[str] = field(default_factory=list)

    async def execute_sql(self, sql: str) -> QueryResult:
        self.sql.append(sql)
        return QueryResult(
            sql=sql, text=self.text, is_error=False, row_count=1, latency_ms=3, truncated=False
        )


@dataclass
class ResolveHarness:
    deps: ResolveDeps
    sink: MemoryTraceSink
    chat: FakeChatClient
    home: FakeHome
    commands: FakeCommandLog
    resolutions: FakeResolutions


def resolve_harness(
    chat: FakeChatClient,
    *,
    settings: Settings | None = None,
    weather: FakeWeather | None = None,
    retrieval: FakeRetrieval | None = None,
    commands: FakeCommandLog | None = None,
    home: FakeHome | None = None,
    knowledge_query: FakeKnowledgeQuery | None = None,
) -> ResolveHarness:
    settings = settings or make_settings()
    sink = MemoryTraceSink()
    home = home or FakeHome()
    commands = commands or FakeCommandLog()
    resolutions = FakeResolutions()
    deps = ResolveDeps(
        chat=chat,
        embedder=FakeEmbeddingClient(dim=8),
        home=home,
        weather=weather or FakeWeather("cloudy"),
        clock=FixedClock(),
        retrieval=retrieval or FakeRetrieval(),
        commands=commands,
        resolutions=resolutions,
        tracer=Tracer(RUN_ID, [sink]),
        settings=settings,
        knowledge_query=knowledge_query,
    )
    return ResolveHarness(deps, sink, chat, home, commands, resolutions)


@dataclass
class FakeTrainKnowledge:
    existing: ExistingSymptom | None = None
    saved: list[tuple[str, uuid.UUID | None, list[ProposedAction], list[ReviewDecision]]] = field(
        default_factory=list
    )

    async def nearest_any(
        self, embedding: Sequence[float], max_distance: float
    ) -> ExistingSymptom | None:
        return self.existing

    async def save_proposals(
        self,
        *,
        symptom_text: str,
        embedding: Sequence[float],
        existing_id: uuid.UUID | None,
        proposals: Sequence[ProposedAction],
        decisions: Sequence[ReviewDecision],
        prompt_version: str | None,
    ) -> tuple[uuid.UUID, list[uuid.UUID]]:
        self.saved.append((symptom_text, existing_id, list(proposals), list(decisions)))
        return existing_id or uuid.UUID(int=50), [uuid.uuid4() for _ in proposals]


def train_deps(
    chat: FakeChatClient, knowledge: FakeTrainKnowledge, sink: MemoryTraceSink
) -> TrainDeps:
    return TrainDeps(
        chat=chat,
        embedder=FakeEmbeddingClient(dim=8),
        knowledge=knowledge,
        tracer=Tracer(RUN_ID, [sink]),
        settings=make_settings(),
    )
