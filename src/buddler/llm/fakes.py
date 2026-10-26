"""Deterministic offline LLM implementations for tests and `LLM_PROVIDER=fake` demos."""

import hashlib
import math
import random
import types
import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, ValidationError

from buddler.llm.budget import estimate_messages_tokens, estimate_tokens
from buddler.llm.ports import (
    ChatMessage,
    EmbeddingResult,
    LLMUsage,
    StructuredOutputError,
    StructuredResult,
)

FAKE_CHAT_MODEL = "fake/chat"
FAKE_EMBED_MODEL = "fake/embed"


def fake_vector(text: str, dim: int) -> list[float]:
    """Unit vector seeded by text hash: same text -> same vector, else ~orthogonal."""
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    values = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


@dataclass
class FakeEmbeddingClient:
    dim: int
    model: str = FAKE_EMBED_MODEL
    calls: list[list[str]] = field(default_factory=list)

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        self.calls.append(list(texts))
        return EmbeddingResult(
            vectors=[fake_vector(text, self.dim) for text in texts],
            usage=LLMUsage(model=self.model, prompt_tokens=sum(map(estimate_tokens, texts))),
            batches=1 if texts else 0,
        )


_SCALAR_PLACEHOLDERS: dict[type, object] = {str: "fake", bool: False, int: 0, float: 0.0}


def _placeholder_value(annotation: object) -> object:
    origin = typing.get_origin(annotation)
    if origin in (list, set, tuple, dict):
        return origin()
    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(annotation)
        return None if type(None) in args else _placeholder_value(args[0])
    if origin is typing.Literal:
        return typing.get_args(annotation)[0]
    if isinstance(annotation, type):
        if issubclass(annotation, Enum):
            return next(iter(annotation))
        if issubclass(annotation, BaseModel):
            return placeholder(annotation)
        if annotation in _SCALAR_PLACEHOLDERS:
            return _SCALAR_PLACEHOLDERS[annotation]
    raise StructuredOutputError(
        f"No placeholder for {annotation!r}; give FakeChatClient a response."
    )


def placeholder[T: BaseModel](schema: type[T]) -> T:
    """Minimal valid instance: defaults where declared, type-based placeholders otherwise."""
    try:
        return schema()
    except ValidationError:
        values = {
            name: _placeholder_value(info.annotation)
            for name, info in schema.model_fields.items()
            if info.is_required()
        }
        return schema.model_validate(values)


Responder = Callable[[Sequence[ChatMessage], type[BaseModel]], BaseModel]


@dataclass
class FakeChatClient:
    """Answers from a queue of canned values, else a responder, else a schema placeholder."""

    responses: list[BaseModel] = field(default_factory=list)
    responder: Responder | None = None
    model: str = FAKE_CHAT_MODEL
    calls: list[list[ChatMessage]] = field(default_factory=list)

    async def complete_structured[T: BaseModel](
        self,
        messages: Sequence[ChatMessage],
        schema: type[T],
        *,
        max_tokens: int | None = None,
    ) -> StructuredResult[T]:
        self.calls.append(list(messages))
        raw = self._next(messages, schema)
        value = schema.model_validate(raw.model_dump())
        usage = LLMUsage(
            model=self.model,
            prompt_tokens=estimate_messages_tokens(messages),
            completion_tokens=estimate_tokens(value.model_dump_json()),
        )
        return StructuredResult(value=value, usage=usage, attempts=1)

    def _next(self, messages: Sequence[ChatMessage], schema: type[BaseModel]) -> BaseModel:
        if self.responses:
            return self.responses.pop(0)
        if self.responder is not None:
            return self.responder(messages, schema)
        return placeholder(schema)
