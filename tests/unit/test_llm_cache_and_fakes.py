import math
from collections.abc import Mapping, Sequence

import pytest
from pydantic import BaseModel, SecretStr

from buddler.config import Settings
from buddler.llm.cache import CachedEmbeddingClient, text_hash
from buddler.llm.factory import build_chat_client, build_embedding_client
from buddler.llm.fakes import FakeChatClient, FakeEmbeddingClient, fake_vector, placeholder
from buddler.llm.ports import ChatMessage, LLMError

DIM = 8


class MemoryStore:
    def __init__(self) -> None:
        self.data: dict[tuple[str, str], list[float]] = {}

    async def get_many(self, model: str, text_hashes: Sequence[str]) -> dict[str, list[float]]:
        return {h: self.data[(model, h)] for h in text_hashes if (model, h) in self.data}

    async def put_many(self, model: str, vectors: Mapping[str, list[float]]) -> None:
        for h, vector in vectors.items():
            self.data[(model, h)] = vector


def test_fake_vector_is_deterministic_unit_vector() -> None:
    first, again, other = fake_vector("a", DIM), fake_vector("a", DIM), fake_vector("b", DIM)

    assert first == again != other
    assert math.isclose(sum(v * v for v in first), 1.0)


async def test_cache_only_embeds_unseen_texts_once() -> None:
    inner = FakeEmbeddingClient(dim=DIM)
    store = MemoryStore()
    cached = CachedEmbeddingClient(inner, store)

    first = await cached.embed(["a", "b", "a"])
    second = await cached.embed(["b", "c"])

    assert inner.calls == [["a", "b"], ["c"]]
    assert first.vectors == [fake_vector(t, DIM) for t in ["a", "b", "a"]]
    assert (first.cache_hits, second.cache_hits) == (1, 1)
    assert (inner.model, text_hash("a")) in store.data


async def test_fully_cached_request_reports_no_usage() -> None:
    cached = CachedEmbeddingClient(FakeEmbeddingClient(dim=DIM), MemoryStore())
    await cached.embed(["a"])

    result = await cached.embed(["a"])

    assert result.usage is None
    assert result.cache_hits == 1


class Verdict(BaseModel):
    anomalous: bool = False


async def test_fake_chat_uses_queue_then_defaults() -> None:
    chat = FakeChatClient(responses=[Verdict(anomalous=True)])
    messages = [ChatMessage(role="user", content="x")]

    first = await chat.complete_structured(messages, Verdict)
    second = await chat.complete_structured(messages, Verdict)

    assert (first.value.anomalous, second.value.anomalous) == (True, False)
    assert len(chat.calls) == 2


class Nested(BaseModel):
    label: str


class Strict(BaseModel):
    flag: bool
    score: float
    tags: list[str]
    note: str | None
    nested: Nested


def test_placeholder_fills_required_fields() -> None:
    assert placeholder(Strict) == Strict(
        flag=False, score=0.0, tags=[], note=None, nested=Nested(label="fake")
    )


def test_factory_requires_api_key_for_real_provider() -> None:
    settings = Settings(_env_file=None, llm_provider="openai", llm_api_key=SecretStr(""))

    with pytest.raises(LLMError, match="LLM_API_KEY"):
        build_chat_client(settings)


def test_factory_builds_fakes_offline() -> None:
    settings = Settings(_env_file=None, llm_provider="fake")

    assert isinstance(build_chat_client(settings), FakeChatClient)
    assert isinstance(build_embedding_client(settings, MemoryStore()), CachedEmbeddingClient)
