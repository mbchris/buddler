"""OpenAI SDK v3 uses httpx2 (respx can't patch it): requests go to an in-memory MockTransport."""

import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel

from buddler.llm.openai_client import (
    OpenAIChatClient,
    OpenAIEmbeddingClient,
    response_json_schema,
)
from buddler.llm.ports import (
    ChatMessage,
    EmbeddingShapeError,
    LLMError,
    StructuredOutputError,
    TokenBudgetExceededError,
)

BASE_URL = "https://router.test/v1"
DIM = 4
CHAT_PATH = "/v1/chat/completions"
EMBED_PATH = "/v1/embeddings"

Handler = Callable[[httpx2.Request], httpx2.Response]


class FakeApi:
    """Routes by path; each route serves a queue of responses (last one repeats) or a handler."""

    def __init__(self) -> None:
        self.routes: dict[str, list[httpx2.Response] | Handler] = {}
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        route = self.routes[request.url.path]
        if callable(route):
            return route(request)
        return route.pop(0) if len(route) > 1 else route[0]

    def bodies(self, path: str) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if r.url.path == path]

    def client(self) -> AsyncOpenAI:
        http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(self))
        return AsyncOpenAI(
            base_url=BASE_URL, api_key="test-key", max_retries=0, http_client=http_client
        )


@pytest.fixture
def api() -> FakeApi:
    return FakeApi()


class Finding(BaseModel):
    title: str
    severity: str


def _chat_response(
    content: str, prompt_tokens: int = 10, completion_tokens: int = 5
) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "id": "cmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "openai/gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        },
    )


def _embedding_handler(dim: int = DIM) -> Handler:
    def handle(request: httpx2.Request) -> httpx2.Response:
        inputs: list[str] = json.loads(request.content)["input"]
        data = [
            {"object": "embedding", "index": i, "embedding": [float(len(text))] * dim}
            for i, text in enumerate(inputs)
        ]
        # Out of order on purpose: client must sort by index.
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "model": "openai/text-embedding-3-small",
                "data": list(reversed(data)),
                "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)},
            },
        )

    return handle


MESSAGES = [ChatMessage(role="system", content="Classify."), ChatMessage(role="user", content="x")]
VALID = '{"title": "DB timeout", "severity": "high"}'


async def test_structured_output_json_schema_mode(api: FakeApi) -> None:
    api.routes[CHAT_PATH] = [_chat_response(VALID)]
    chat = OpenAIChatClient(api.client(), model="openai/gpt-4o-mini")

    result = await chat.complete_structured(MESSAGES, Finding)

    assert result.value == Finding(title="DB timeout", severity="high")
    assert (result.attempts, result.usage.prompt_tokens, result.usage.completion_tokens) == (
        1,
        10,
        5,
    )
    body = api.bodies(CHAT_PATH)[-1]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["name"] == "Finding"
    assert body["temperature"] == 0.0
    assert api.requests[-1].headers["authorization"] == "Bearer test-key"


async def test_json_object_mode_puts_schema_in_prompt(api: FakeApi) -> None:
    api.routes[CHAT_PATH] = [_chat_response(VALID)]
    chat = OpenAIChatClient(api.client(), model="m", mode="json_object")

    await chat.complete_structured(MESSAGES, Finding)

    body = api.bodies(CHAT_PATH)[-1]
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0]["role"] == "system"
    assert '"title"' in body["messages"][0]["content"]


async def test_markdown_fenced_json_is_accepted(api: FakeApi) -> None:
    api.routes[CHAT_PATH] = [_chat_response(f"```json\n{VALID}\n```")]

    result = await OpenAIChatClient(api.client(), model="m").complete_structured(MESSAGES, Finding)

    assert result.value.title == "DB timeout"


async def test_invalid_output_is_retried_with_feedback(api: FakeApi) -> None:
    api.routes[CHAT_PATH] = [_chat_response('{"title": 1}'), _chat_response(VALID)]

    result = await OpenAIChatClient(api.client(), model="m").complete_structured(MESSAGES, Finding)

    assert result.attempts == 2
    assert result.usage.prompt_tokens == 20
    retry_messages = api.bodies(CHAT_PATH)[-1]["messages"]
    assert retry_messages[-2] == {"role": "assistant", "content": '{"title": 1}'}
    assert retry_messages[-1]["content"].startswith("Invalid output:")


async def test_gives_up_after_bounded_retries(api: FakeApi) -> None:
    api.routes[CHAT_PATH] = [_chat_response("nope")]
    chat = OpenAIChatClient(api.client(), model="m", structured_retries=1)

    with pytest.raises(StructuredOutputError, match="after 2 attempts"):
        await chat.complete_structured(MESSAGES, Finding)

    assert len(api.requests) == 2


async def test_budget_is_checked_before_calling_api(api: FakeApi) -> None:
    chat = OpenAIChatClient(api.client(), model="m", max_input_tokens=10)

    with pytest.raises(TokenBudgetExceededError):
        await chat.complete_structured([ChatMessage(role="user", content="x" * 400)], Finding)

    assert api.requests == []


async def test_http_error_becomes_llm_error(api: FakeApi) -> None:
    api.routes[CHAT_PATH] = [httpx2.Response(500, json={"error": {"message": "boom"}})]

    with pytest.raises(LLMError, match="Chat completion failed"):
        await OpenAIChatClient(api.client(), model="m").complete_structured(MESSAGES, Finding)


async def test_embeddings_are_batched_and_ordered(api: FakeApi) -> None:
    api.routes[EMBED_PATH] = _embedding_handler()
    client = OpenAIEmbeddingClient(api.client(), model="e", dim=DIM, batch_size=2)

    result = await client.embed(["a", "bb", "ccc", "dddd", "eeeee"])

    assert [vector[0] for vector in result.vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert (len(api.requests), result.batches) == (3, 3)
    assert result.usage is not None and result.usage.prompt_tokens == 5
    body = api.bodies(EMBED_PATH)[0]
    assert (body["dimensions"], body["encoding_format"]) == (DIM, "float")


async def test_long_inputs_are_truncated_and_counted(api: FakeApi) -> None:
    api.routes[EMBED_PATH] = _embedding_handler()
    client = OpenAIEmbeddingClient(api.client(), model="e", dim=DIM, max_input_chars=3)

    result = await client.embed(["ab", "abcdef"])

    assert result.truncated_inputs == 1
    assert api.bodies(EMBED_PATH)[-1]["input"] == ["ab", "abc"]


async def test_wrong_dimension_is_rejected(api: FakeApi) -> None:
    api.routes[EMBED_PATH] = _embedding_handler(dim=DIM + 1)
    client = OpenAIEmbeddingClient(api.client(), model="e", dim=DIM)

    with pytest.raises(EmbeddingShapeError):
        await client.embed(["a"])


@pytest.mark.parametrize("texts", [["ok", ""], ["   "]])
async def test_blank_embedding_inputs_are_rejected(api: FakeApi, texts: list[str]) -> None:
    client = OpenAIEmbeddingClient(api.client(), model="e", dim=DIM)

    with pytest.raises(ValueError, match="non-blank"):
        await client.embed(texts)


async def test_empty_embedding_request_makes_no_call(api: FakeApi) -> None:
    result = await OpenAIEmbeddingClient(api.client(), model="e", dim=DIM).embed([])

    assert (result.vectors, result.usage) == ([], None)
    assert api.requests == []


class Inner(BaseModel):
    label: str


class Outer(BaseModel):
    inner: Inner
    items: list[Inner]
    note: str | None = None


def test_response_schema_closes_every_object() -> None:
    schema = response_json_schema(Outer)

    assert schema["additionalProperties"] is False
    defs = schema["$defs"]
    assert isinstance(defs, dict)
    assert defs["Inner"]["additionalProperties"] is False
    assert schema["required"] == ["inner", "items", "note"]
