"""Embedding cache decorator: only unseen texts reach the API."""

import hashlib
from collections.abc import Mapping, Sequence
from typing import Protocol

from buddler.llm.ports import EmbeddingClient, EmbeddingResult


class EmbeddingStore(Protocol):
    async def get_many(self, model: str, text_hashes: Sequence[str]) -> dict[str, list[float]]: ...

    async def put_many(self, model: str, vectors: Mapping[str, list[float]]) -> None: ...


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CachedEmbeddingClient:
    def __init__(self, inner: EmbeddingClient, store: EmbeddingStore) -> None:
        self._inner = inner
        self._store = store
        self.model = inner.model
        self.dim = inner.dim

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        hashes = [text_hash(text) for text in texts]
        known = await self._store.get_many(self.model, list(set(hashes)))
        # Deduplicate misses so repeated texts in one request cost one embedding.
        missing = {h: text for h, text in zip(hashes, texts, strict=True) if h not in known}
        result = await self._inner.embed(list(missing.values())) if missing else None
        if result is not None:
            fresh = dict(zip(missing.keys(), result.vectors, strict=True))
            await self._store.put_many(self.model, fresh)
            known.update(fresh)
        return EmbeddingResult(
            vectors=[known[h] for h in hashes],
            usage=result.usage if result else None,
            truncated_inputs=result.truncated_inputs if result else 0,
            cache_hits=len(texts) - len(missing),
            batches=result.batches if result else 0,
        )
