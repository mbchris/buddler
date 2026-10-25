"""REST API end-to-end: real PostgreSQL, fake LLM/home/weather, no MCP."""

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr
from sqlalchemy import delete

from buddler.agents.graphs.resolve import RESOLVE_NODE_ORDER
from buddler.api.main import create_app
from buddler.api.run_manager import RunManager
from buddler.config import Settings
from buddler.db import session_scope
from buddler.llm.fakes import FakeEmbeddingClient
from buddler.models import DeviceCommand, EmbeddingCacheEntry, Run, Symptom
from buddler.schemas.knowledge import SeedFile
from buddler.services.knowledge_service import KnowledgeService

API_KEY = "test-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}
# The fake LLM answers with placeholders: its symptom is "fake", so seed exactly that.
SEED = SeedFile.model_validate(
    {
        "symptoms": [
            {
                "text": "fake",
                "actions": [{"kind": "advice", "title": "Tea", "instruction": "Drink tea."}],
            }
        ]
    }
)


@pytest.fixture
def settings(migrated_database: str) -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="fake",
        homematic_provider="fake",
        weather_provider="fake",
        default_room="living room",
        database_url=migrated_database,
        api_key=SecretStr(API_KEY),
        sse_poll_interval_s=0.01,
    )


@pytest.fixture
async def app(settings: Settings) -> AsyncIterator[FastAPI]:
    app = create_app(settings)
    yield app
    await app.state.run_manager.shutdown()
    async with app.state.session_factory() as session:
        for model in (DeviceCommand, Run, Symptom, EmbeddingCacheEntry):
            await session.execute(delete(model))
        await session.commit()


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=AUTH
    ) as client:
        yield client


async def _seed(app: FastAPI, settings: Settings) -> None:
    async with session_scope(app.state.session_factory) as session:
        await KnowledgeService(session).import_seed(SEED, FakeEmbeddingClient(settings.embed_dim))


def _manager(app: FastAPI) -> RunManager:
    manager: RunManager = app.state.run_manager
    return manager


async def _run(app: FastAPI, client: AsyncClient, **body: object) -> dict[str, Any]:
    accepted = await client.post("/runs", json=body)
    assert accepted.status_code == 202, accepted.text
    run_id = accepted.json()["run_id"]
    await _manager(app).join(uuid.UUID(run_id))
    detail = await client.get(f"/runs/{run_id}")
    assert detail.status_code == 200
    result: dict[str, Any] = detail.json()
    return result


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}])
async def test_api_requires_key(app: FastAPI, headers: dict[str, str]) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as anon:
        protected = await anon.get("/knowledge/symptoms", headers=headers)
        health = await anon.get("/health")

    assert protected.status_code == 401
    assert health.status_code == 200


async def test_train_then_review_via_api(app: FastAPI, client: AsyncClient) -> None:
    detail = await _run(app, client, kind="train", symptom="feeling cold", review_mode="defer")

    assert detail["run"]["status"] == "succeeded"
    assert detail["run"]["subject"] == "feeling cold"
    [symptom] = (await client.get("/knowledge/symptoms", params={"status": "proposed"})).json()
    reviewed = await client.patch(
        f"/knowledge/symptoms/{symptom['id']}", json={"status": "confirmed"}
    )
    assert reviewed.status_code == 200
    assert reviewed.json()["status"] == "confirmed" and reviewed.json()["reviewed_at"]


@pytest.mark.parametrize(
    "body",
    [
        {"kind": "train", "symptom": "x", "review_mode": "interactive"},
        {"kind": "resolve"},
        {"kind": "deploy"},
    ],
)
async def test_run_request_validation(client: AsyncClient, body: dict[str, str]) -> None:
    assert (await client.post("/runs", json=body)).status_code == 422


async def test_unknown_ids_return_404(client: AsyncClient) -> None:
    missing = str(uuid.uuid4())

    responses = [
        await client.get(f"/runs/{missing}"),
        await client.get(f"/runs/{missing}/events"),
        await client.get(f"/runs/{missing}/stream"),
        await client.patch(f"/knowledge/symptoms/{missing}", json={"status": "rejected"}),
        await client.patch(f"/knowledge/actions/{missing}", json={"status": "rejected"}),
    ]

    assert [r.status_code for r in responses] == [404] * len(responses)


def _sse(text: str) -> list[dict[str, str]]:
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields:
            events.append(fields)
    return events


async def test_resolve_run_detail_events_and_stream(
    app: FastAPI, client: AsyncClient, settings: Settings
) -> None:
    await _seed(app, settings)

    detail = await _run(
        app,
        client,
        kind="resolve",
        situation="I'm cold",
        at="2026-11-10T02:00:00",
        use_mcp=False,
    )

    run_id = detail["run"]["id"]
    assert detail["run"]["status"] == "succeeded", detail["run"]["error"]
    resolution = detail["resolution"]
    assert resolution["room"] == "living room"
    assert resolution["observations"]["climate"]["setpoint_c"] == 20.0
    assert resolution["prompt_versions"].keys() == {"understand", "agent", "decide"}
    events = (await client.get(f"/runs/{run_id}/events")).json()
    nodes = [e["node"] for e in events if e["type"] == "node_start"]
    assert nodes == list(RESOLVE_NODE_ORDER)
    candidates = next(e for e in events if e["node"] == "evaluate" and e["type"] == "node_end")
    assert candidates["payload"]["candidates"][0]["title"] == "Tea"

    replay = _sse((await client.get(f"/runs/{run_id}/stream")).text)
    resumed = _sse(
        (await client.get(f"/runs/{run_id}/stream", headers={"Last-Event-ID": "5"})).text
    )

    assert [e["event"] for e in replay] == ["trace"] * len(events) + ["end"]
    assert json.loads(replay[-1]["data"])["status"] == "succeeded"
    assert resumed[0]["id"] == "6"
    assert len(resumed) == len(events) - 5 + 1


async def test_runs_list_home_and_graphs(app: FastAPI, client: AsyncClient) -> None:
    await _run(app, client, kind="train", symptom="feeling warm", review_mode="defer")

    runs = (await client.get("/runs")).json()
    rooms = (await client.get("/home/rooms")).json()
    graph = (await client.get("/graphs/resolve")).json()

    assert [r["kind"] for r in runs] == ["train"]
    assert {r["room"] for r in rooms} == {"living room", "bedroom", "office"}
    assert "evaluate" in graph["mermaid"] and len(graph["steps"]) == len(RESOLVE_NODE_ORDER)
