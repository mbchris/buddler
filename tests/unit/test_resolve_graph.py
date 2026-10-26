"""Resolve workflow end to end with fakes: the AGENTS.md §1 scenarios and the actuation guard."""

from collections.abc import Sequence
from datetime import datetime, timedelta

import pytest
from pydantic import BaseModel

from buddler.agents.graphs.resolve import RESOLVE_NODE_ORDER, build_resolve_graph
from buddler.agents.nodes.resolve import match_room
from buddler.agents.schemas import (
    AdviceDraft,
    AgentStep,
    ChosenActionDraft,
    DecisionDraft,
    UnderstandingDraft,
)
from buddler.agents.state import ActionOutcome, Resolution, ResolveState, RoomSource
from buddler.llm.fakes import FakeChatClient, placeholder
from buddler.llm.ports import ChatMessage, StructuredOutputError
from buddler.models.enums import CommandStatus, TraceEventType
from buddler.weather.fake import FakeWeather
from tests.unit.workflow_fakes import (
    FakeCommandLog,
    FakeKnowledgeQuery,
    FixedClock,
    ResolveHarness,
    make_settings,
    resolve_harness,
)

FINISH = AgentStep(thought="Enough.", action="finish", notes="Living room is cold (O3).")


def cold(room: str | None = None) -> UnderstandingDraft:
    return UnderstandingDraft(symptom="feeling cold", room_mention=room, urgency="normal")


def decision(*refs: str, advice: str | None = None) -> DecisionDraft:
    return DecisionDraft(
        reply="Done.",
        summary="Decided.",
        chosen=[ChosenActionDraft(ref=r, rationale="fits", citations=[r, "O1"]) for r in refs],
        advice=[AdviceDraft(text=advice, citations=[])] if advice else [],
        confidence=0.8,
    )


def chat(*responses: BaseModel) -> FakeChatClient:
    return FakeChatClient(responses=list(responses))


async def run(harness: ResolveHarness, at: str, *, dry_run: bool = False) -> dict[str, object]:
    hour, minute = map(int, at.split(":"))
    inputs: ResolveState = {
        "situation": "I'm feeling cold",
        "at": datetime(2026, 10, 25, hour, minute),
        "dry_run": dry_run,
    }
    state: dict[str, object] = await build_resolve_graph(harness.deps).ainvoke(inputs)
    return state


def resolution(state: dict[str, object]) -> Resolution:
    value = state["resolution"]
    assert isinstance(value, Resolution)
    return value


async def test_scenario_a_night_rejects_heating_and_advises() -> None:
    harness = resolve_harness(chat(cold(), FINISH, decision("A1", "A2")))

    state = await run(harness, "02:00")

    candidates = {c.ref: c for c in state["candidates"]}  # type: ignore[attr-defined]
    assert not candidates["A1"].applicable
    assert "time_window" in candidates["A1"].reason
    assert candidates["A2"].applicable
    outcomes = {a.ref: a.outcome for a in resolution(state).actions}
    assert outcomes == {"A1": ActionOutcome.REJECTED, "A2": ActionOutcome.ADVISED}
    assert harness.home.writes == []
    assert [r.status for r in harness.commands.records] == [CommandStatus.REJECTED]


async def test_scenario_b_cloudy_afternoon_raises_heating() -> None:
    harness = resolve_harness(chat(cold("Wohnzimmer"), FINISH, decision("A1")))

    state = await run(harness, "13:00")

    assert state["room"] == "living room"
    assert state["room_source"] is RoomSource.MENTIONED
    [action] = resolution(state).actions
    assert action.outcome is ActionOutcome.APPLIED
    assert harness.home.writes == [("living room", 21.0)]
    [record] = harness.commands.records
    assert (record.old_value, record.new_value, record.status) == (20.0, 21.0, "applied")
    home_calls = [e for e in harness.sink.events if e.type is TraceEventType.HOME_CALL]
    assert any("set living room" in e.summary for e in home_calls)
    assert [s for s in harness.resolutions.saved] == [resolution(state)]


async def test_sunny_afternoon_blocks_heating_and_offers_sun_advice() -> None:
    harness = resolve_harness(chat(cold(), FINISH, decision("A3")), weather=FakeWeather("sunny"))

    state = await run(harness, "13:00")

    candidates = {c.ref: c for c in state["candidates"]}  # type: ignore[attr-defined]
    assert not candidates["A1"].applicable and "sun_hours" in candidates["A1"].reason
    assert candidates["A3"].applicable
    assert resolution(state).actions[0].outcome is ActionOutcome.ADVISED


async def test_dry_run_logs_without_writing() -> None:
    harness = resolve_harness(chat(cold(), FINISH, decision("A1")))

    state = await run(harness, "13:00", dry_run=True)

    assert resolution(state).actions[0].outcome is ActionOutcome.DRY_RUN
    assert harness.home.writes == []
    assert harness.commands.records[0].status is CommandStatus.DRY_RUN


async def test_cooldown_rejects_repeated_write() -> None:
    recently = FixedClock().value - timedelta(minutes=10)
    harness = resolve_harness(
        chat(cold(), FINISH, decision("A1")), commands=FakeCommandLog(last_applied=recently)
    )

    state = await run(harness, "13:00")

    [action] = resolution(state).actions
    assert action.outcome is ActionOutcome.REJECTED and "cooldown" in action.detail
    assert harness.home.writes == []


async def test_room_outside_allowlist_is_never_written() -> None:
    harness = resolve_harness(
        chat(cold(), FINISH, decision("A1")), settings=make_settings(actuation_rooms=[])
    )

    state = await run(harness, "13:00")

    assert "ACTUATION_ROOMS" in resolution(state).actions[0].detail
    assert harness.home.writes == []


async def test_unknown_refs_are_dropped_and_uncited_advice_flagged() -> None:
    harness = resolve_harness(chat(cold(), FINISH, decision("A9", advice="Drink tea.")))

    state = await run(harness, "02:00")

    res = resolution(state)
    assert res.actions == [] and res.invalid_refs == ["A9"]
    assert res.advice[0].uncited and res.uncited_count == 1


async def test_agent_uses_query_and_read_room_tools_within_budget() -> None:
    query = FakeKnowledgeQuery()
    steps = [
        AgentStep(thought="History?", action="query", sql="SELECT 1"),
        AgentStep(thought="Bedroom?", action="read_room", room="Schlafzimmer"),
        AgentStep(thought="Again", action="query", sql="SELECT 2"),
    ]
    harness = resolve_harness(chat(cold(), *steps, decision()), knowledge_query=query)

    state = await run(harness, "13:00")

    agent = state["agent"]
    assert [c.tool for c in agent.tool_calls] == ["query", "read_room"]  # type: ignore[attr-defined]
    assert agent.tool_calls[1].argument == "bedroom"  # type: ignore[attr-defined]
    assert agent.stopped_reason == "max_steps"  # type: ignore[attr-defined]
    assert query.sql == ["SELECT 1"]
    assert any(e.type is TraceEventType.MCP_CALL for e in harness.sink.events)


async def test_understanding_failure_falls_back_to_sentence() -> None:
    def responder(messages: Sequence[ChatMessage], schema: type[BaseModel]) -> BaseModel:
        if schema is UnderstandingDraft:
            raise StructuredOutputError("bad output")
        return FINISH if schema is AgentStep else placeholder(schema)

    harness = resolve_harness(FakeChatClient(responder=responder))

    state = await run(harness, "13:00")

    understanding = state["understanding"]
    assert understanding.symptom == "I'm feeling cold"  # type: ignore[attr-defined]
    assert not understanding.from_llm  # type: ignore[attr-defined]
    assert any(e.type is TraceEventType.ERROR for e in harness.sink.events)


async def test_every_node_emits_a_span() -> None:
    harness = resolve_harness(chat(cold(), FINISH, decision()))

    await run(harness, "13:00")

    ended = [e.node for e in harness.sink.events if e.type is TraceEventType.NODE_END]
    assert ended == list(RESOLVE_NODE_ORDER)


async def test_simulated_time_is_local_and_flagged() -> None:
    harness = resolve_harness(chat(cold(), FINISH, decision()))

    state = await run(harness, "02:00")

    observations = state["observations"]
    assert observations.simulated_time  # type: ignore[attr-defined]
    local = observations.local_time  # type: ignore[attr-defined]
    assert (local.hour, local.minute) == (2, 0)
    assert str(local.tzinfo) == "Europe/Berlin"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("living room", "living room"),
        ("Wohnzimmer", "living room"),
        ("the living-room", "living room"),
        ("Büro", "office"),
        ("garage", None),
    ],
)
def test_match_room(text: str, expected: str | None) -> None:
    rooms = {"living room": ["wohnzimmer"], "office": ["büro"], "bedroom": []}

    assert match_room(text, rooms) == expected
