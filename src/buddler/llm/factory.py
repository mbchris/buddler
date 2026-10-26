"""Build LLM clients from settings; the only place that knows concrete implementations."""

from buddler.config import Settings
from buddler.llm.cache import CachedEmbeddingClient, EmbeddingStore
from buddler.llm.fakes import FakeChatClient, FakeEmbeddingClient
from buddler.llm.openai_client import (
    OpenAIChatClient,
    OpenAIEmbeddingClient,
    create_openai_client,
)
from buddler.llm.ports import ChatClient, EmbeddingClient, LLMError


def _require_api_key(settings: Settings) -> None:
    if not settings.llm_api_key.get_secret_value():
        raise LLMError("LLM_API_KEY is empty; set it in .env or use LLM_PROVIDER=fake.")


def build_chat_client(settings: Settings) -> ChatClient:
    if settings.llm_provider == "fake":
        return FakeChatClient()
    _require_api_key(settings)
    return OpenAIChatClient.from_settings(settings, create_openai_client(settings))


def build_embedding_client(
    settings: Settings, store: EmbeddingStore | None = None
) -> EmbeddingClient:
    client: EmbeddingClient
    if settings.llm_provider == "fake":
        client = FakeEmbeddingClient(dim=settings.embed_dim)
    else:
        _require_api_key(settings)
        client = OpenAIEmbeddingClient.from_settings(settings, create_openai_client(settings))
    return CachedEmbeddingClient(client, store) if store is not None else client
