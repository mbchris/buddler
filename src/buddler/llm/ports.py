"""LLM ports: the only way the rest of the app talks to chat/embedding models (AGENTS.md §2)."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel

Role = Literal["system", "user", "assistant"]


class LLMError(RuntimeError):
    pass


class TokenBudgetExceededError(LLMError):
    pass


class StructuredOutputError(LLMError):
    """Model output did not validate against the schema after all retries."""


class EmbeddingShapeError(LLMError):
    pass


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class LLMUsage:
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0

    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        return LLMUsage(
            model=self.model,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            latency_ms=self.latency_ms + other.latency_ms,
        )


@dataclass(frozen=True)
class StructuredResult[T: BaseModel]:
    value: T
    usage: LLMUsage
    attempts: int


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: list[list[float]]
    # None when every vector came from the cache (no API call).
    usage: LLMUsage | None
    truncated_inputs: int = 0
    cache_hits: int = 0
    batches: int = 0


class ChatClient(Protocol):
    model: str

    async def complete_structured[T: BaseModel](
        self,
        messages: Sequence[ChatMessage],
        schema: type[T],
        *,
        max_tokens: int | None = None,
    ) -> StructuredResult[T]: ...


class EmbeddingClient(Protocol):
    model: str
    dim: int

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult: ...
