"""OpenAI-compatible chat + embedding clients (Requesty router or any `/v1` endpoint)."""

import asyncio
import json
import logging
import re
import time
from collections.abc import Sequence
from typing import Literal

from openai import APIError, AsyncOpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.shared_params import ResponseFormatJSONObject, ResponseFormatJSONSchema
from pydantic import BaseModel, ValidationError

from buddler.config import Settings
from buddler.llm.budget import ensure_within_budget, truncate_for_embedding
from buddler.llm.ports import (
    ChatMessage,
    EmbeddingResult,
    EmbeddingShapeError,
    LLMError,
    LLMUsage,
    StructuredOutputError,
    StructuredResult,
)

logger = logging.getLogger(__name__)

StructuredMode = Literal["json_schema", "json_object"]
# Some models wrap JSON in markdown fences despite response_format.
_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def create_openai_client(settings: Settings) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _strip_fences(content: str) -> str:
    match = _JSON_FENCE.match(content)
    return match.group(1) if match else content


def _closed(node: object) -> object:
    """OpenAI json_schema needs `additionalProperties: false` and every property `required`.

    Optional fields stay nullable (`anyOf: [..., null]`), so validation is unchanged.
    """
    if isinstance(node, dict):
        closed = {key: _closed(value) for key, value in node.items()}
        if closed.get("type") == "object" or "properties" in closed:
            closed["additionalProperties"] = False
            properties = closed.get("properties")
            if isinstance(properties, dict):
                closed["required"] = list(properties)
        return closed
    if isinstance(node, list):
        return [_closed(item) for item in node]
    return node


def response_json_schema(schema: type[BaseModel]) -> dict[str, object]:
    closed = _closed(schema.model_json_schema())
    assert isinstance(closed, dict)  # top-level pydantic schema is always an object
    return closed


def _schema_instruction(schema: type[BaseModel]) -> ChatMessage:
    return ChatMessage(
        role="system",
        content=(
            "Respond with a single JSON object only, no prose, matching this JSON schema:\n"
            + json.dumps(response_json_schema(schema))
        ),
    )


def _to_param(message: ChatMessage) -> ChatCompletionMessageParam:
    match message.role:
        case "system":
            return ChatCompletionSystemMessageParam(role="system", content=message.content)
        case "user":
            return ChatCompletionUserMessageParam(role="user", content=message.content)
        case "assistant":
            return ChatCompletionAssistantMessageParam(role="assistant", content=message.content)


class OpenAIChatClient:
    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str,
        mode: StructuredMode = "json_schema",
        temperature: float = 0.0,
        max_tokens: int = 4000,
        max_input_tokens: int = 16000,
        structured_retries: int = 2,
    ) -> None:
        self._client = client
        self.model = model
        self._mode = mode
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_input_tokens = max_input_tokens
        self._structured_retries = structured_retries

    @classmethod
    def from_settings(cls, settings: Settings, client: AsyncOpenAI) -> "OpenAIChatClient":
        return cls(
            client,
            model=settings.llm_chat_model,
            mode=settings.llm_structured_mode,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens_per_call,
            max_input_tokens=settings.llm_max_input_tokens,
            structured_retries=settings.llm_structured_retries,
        )

    async def complete_structured[T: BaseModel](
        self,
        messages: Sequence[ChatMessage],
        schema: type[T],
        *,
        max_tokens: int | None = None,
    ) -> StructuredResult[T]:
        conversation = list(messages)
        if self._mode == "json_object":
            conversation.insert(0, _schema_instruction(schema))
        usage = LLMUsage(model=self.model)
        last_error = ""
        for attempt in range(1, self._structured_retries + 2):
            ensure_within_budget(conversation, self._max_input_tokens)
            content, call_usage = await self._complete(conversation, schema, max_tokens)
            usage += call_usage
            try:
                value = schema.model_validate_json(_strip_fences(content))
            except ValidationError as err:
                last_error = str(err)
                logger.warning(
                    "Invalid structured output (attempt %d, %d errors)", attempt, err.error_count()
                )
                conversation += [
                    ChatMessage(role="assistant", content=content),
                    ChatMessage(
                        role="user",
                        content=f"Invalid output: {last_error}\nReturn only corrected JSON.",
                    ),
                ]
                continue
            return StructuredResult(value=value, usage=usage, attempts=attempt)
        msg = f"No valid {schema.__name__} after {attempt} attempts: {last_error}"
        raise StructuredOutputError(msg)

    async def _complete(
        self,
        messages: Sequence[ChatMessage],
        schema: type[BaseModel],
        max_tokens: int | None,
    ) -> tuple[str, LLMUsage]:
        params = [_to_param(message) for message in messages]
        started = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=params,
                temperature=self._temperature,
                max_tokens=max_tokens or self._max_tokens,
                response_format=self._response_format(schema),
            )
        except APIError as err:
            raise LLMError(f"Chat completion failed: {err}") from err
        if not response.choices:
            raise LLMError("Chat completion returned no choices.")
        choice = response.choices[0]
        if choice.finish_reason == "length":
            logger.warning("Completion hit max_tokens; output may be truncated")
        usage = LLMUsage(
            model=response.model or self.model,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            latency_ms=_elapsed_ms(started),
        )
        return choice.message.content or "", usage

    def _response_format(self, schema: type[BaseModel]) -> ResponseFormat:
        if self._mode == "json_object":
            return ResponseFormatJSONObject(type="json_object")
        return ResponseFormatJSONSchema(
            type="json_schema",
            json_schema={"name": schema.__name__, "schema": response_json_schema(schema)},
        )


class OpenAIEmbeddingClient:
    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str,
        dim: int,
        batch_size: int = 128,
        concurrency: int = 4,
        max_input_chars: int = 8000,
    ) -> None:
        self._client = client
        self.model = model
        self.dim = dim
        self._batch_size = batch_size
        self._semaphore = asyncio.Semaphore(concurrency)
        self._max_input_chars = max_input_chars

    @classmethod
    def from_settings(cls, settings: Settings, client: AsyncOpenAI) -> "OpenAIEmbeddingClient":
        return cls(
            client,
            model=settings.llm_embed_model,
            dim=settings.embed_dim,
            batch_size=settings.embed_batch_size,
            concurrency=settings.embed_concurrency,
            max_input_chars=settings.embed_max_input_chars,
        )

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=[], usage=None)
        if any(not text.strip() for text in texts):
            raise ValueError("Embedding inputs must be non-blank.")
        inputs, truncated = truncate_for_embedding(texts, self._max_input_chars)
        batches = [
            inputs[start : start + self._batch_size]
            for start in range(0, len(inputs), self._batch_size)
        ]
        results = await asyncio.gather(*(self._embed_batch(batch) for batch in batches))
        vectors = [vector for batch_vectors, _ in results for vector in batch_vectors]
        usage = LLMUsage(model=self.model)
        for _, batch_usage in results:
            usage += batch_usage
        return EmbeddingResult(
            vectors=vectors, usage=usage, truncated_inputs=truncated, batches=len(batches)
        )

    async def _embed_batch(self, batch: list[str]) -> tuple[list[list[float]], LLMUsage]:
        async with self._semaphore:
            started = time.perf_counter()
            try:
                response = await self._client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dim,
                    encoding_format="float",
                )
            except APIError as err:
                raise LLMError(f"Embedding request failed: {err}") from err
        items = sorted(response.data, key=lambda item: item.index)
        if len(items) != len(batch):
            raise EmbeddingShapeError(f"Expected {len(batch)} embeddings, got {len(items)}.")
        vectors = [item.embedding for item in items]
        if any(len(vector) != self.dim for vector in vectors):
            raise EmbeddingShapeError(f"Embedding dimension differs from {self.dim}.")
        usage = LLMUsage(
            model=self.model,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            latency_ms=_elapsed_ms(started),
        )
        return vectors, usage
