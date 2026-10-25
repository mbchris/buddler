"""Persistence for cached embeddings."""

from collections.abc import Mapping, Sequence

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from buddler.models.embedding_cache import EmbeddingCacheEntry

# asyncpg caps bind parameters at 32767; 3 columns per row.
BATCH_SIZE = 5000


class EmbeddingCacheRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_many(self, model: str, text_hashes: Sequence[str]) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        for start in range(0, len(text_hashes), BATCH_SIZE):
            query = select(EmbeddingCacheEntry.text_hash, EmbeddingCacheEntry.embedding).where(
                EmbeddingCacheEntry.model == model,
                EmbeddingCacheEntry.text_hash.in_(text_hashes[start : start + BATCH_SIZE]),
            )
            for text_hash, embedding in await self._session.execute(query):
                found[text_hash] = [float(value) for value in embedding]
        return found

    async def put_many(self, model: str, vectors: Mapping[str, list[float]]) -> None:
        rows = [
            {"model": model, "text_hash": text_hash, "embedding": vector}
            for text_hash, vector in vectors.items()
        ]
        for start in range(0, len(rows), BATCH_SIZE):
            statement = insert(EmbeddingCacheEntry).on_conflict_do_nothing()
            await self._session.execute(statement, rows[start : start + BATCH_SIZE])
