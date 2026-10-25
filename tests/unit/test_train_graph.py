import uuid
from datetime import time
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from buddler.agents.graphs.train import TRAIN_NODE_ORDER, build_train_graph
from buddler.agents.schemas import ActionProposalDraft, ProposalResponse
from buddler.agents.state import ExistingSymptom, ReviewMode
from buddler.knowledge.conditions import ActionConditions, ActionParams
from buddler.llm.fakes import FakeChatClient
from buddler.models.enums import ActionKind, KnowledgeStatus, TraceEventType
from buddler.tracing.tracer import MemoryTraceSink
from tests.unit.workflow_fakes import HEAT_UP, FakeTrainKnowledge, train_deps

PROPOSAL = ProposalResponse(
    symptom="room too dark",
    actions=[
        ActionProposalDraft(
            kind=ActionKind.ADVICE,
            title="Open the blinds",
            instruction="Open the blinds.",
            params=ActionParams(),
            conditions=ActionConditions(time_from=time(7), time_to=time(19)),
            priority=10,
        ),
        ActionProposalDraft(
            kind=ActionKind.HEATING_DELTA,
            title="Huge heat boost",
            instruction="Raise by 5 C.",
            params=ActionParams(delta_c=5),
            conditions=ActionConditions(),
            priority=20,
        ),
    ],
)


async def run(
    mode: ReviewMode, knowledge: FakeTrainKnowledge, sink: MemoryTraceSink
) -> dict[str, Any]:
    graph = build_train_graph(train_deps(FakeChatClient(responses=[PROPOSAL]), knowledge, sink))
    state: dict[str, Any] = await graph.ainvoke(
        {"symptom": "it is so dark", "hint": None, "review_mode": mode}
    )
    return state


async def test_auto_review_confirms_and_drops_out_of_bounds_device_action() -> None:
    knowledge, sink = FakeTrainKnowledge(), MemoryTraceSink()

    state = await run(ReviewMode.AUTO_CONFIRM, knowledge, sink)

    assert [p.title for p in state["proposals"]] == ["Open the blinds"]
    [(text, existing_id, _, decisions)] = knowledge.saved
    assert (text, existing_id) == ("room too dark", None)
    assert [d.status for d in decisions] == [KnowledgeStatus.CONFIRMED]
    errors = [e.summary for e in sink.events if e.type is TraceEventType.ERROR]
    assert errors and "exceeds max" in errors[0]
    ended = [e.node for e in sink.events if e.type is TraceEventType.NODE_END]
    assert ended == list(TRAIN_NODE_ORDER)


async def test_existing_symptom_receives_new_actions() -> None:
    existing = ExistingSymptom(
        symptom_id=uuid.UUID(int=7),
        text="dark room",
        status=KnowledgeStatus.CONFIRMED,
        distance=0.05,
        actions=[HEAT_UP],
    )
    knowledge = FakeTrainKnowledge(existing=existing)

    state = await run(ReviewMode.DEFER, knowledge, MemoryTraceSink())

    assert state["saved_symptom_id"] == existing.symptom_id
    [(text, existing_id, _, decisions)] = knowledge.saved
    assert (text, existing_id) == ("dark room", existing.symptom_id)
    assert [d.status for d in decisions] == [KnowledgeStatus.PROPOSED]


async def test_interactive_review_pauses_and_resumes() -> None:
    knowledge, sink = FakeTrainKnowledge(), MemoryTraceSink()
    chat = FakeChatClient(responses=[PROPOSAL])
    graph = build_train_graph(train_deps(chat, knowledge, sink), InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "t1"}}

    paused = await graph.ainvoke(
        {"symptom": "dark", "hint": "advice only", "review_mode": ReviewMode.INTERACTIVE},
        config,
    )
    assert "__interrupt__" in paused and knowledge.saved == []

    await graph.ainvoke(Command(resume=[{"ref": 1, "status": "rejected"}]), config)

    [(_, _, _, decisions)] = knowledge.saved
    assert [d.status for d in decisions] == [KnowledgeStatus.REJECTED]
    # Developer input reaches the LLM only inside delimited data blocks.
    assert "<hint>\nadvice only\n</hint>" in chat.calls[0][1].content
