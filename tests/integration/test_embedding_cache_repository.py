import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from buddler.llm.fakes import fake_vector
from buddler.models import EMBEDDING_DIM
from buddler.repositories.embedding_cache_repository import EmbeddingCacheRepository


async def test_put_and_get_roundtrip_per_model(session: AsyncSession) -> None:
    repo = EmbeddingCacheRepository(session)
    vector = fake_vector("a", EMBEDDING_DIM)

    await repo.put_many("model-a", {"h1": vector})

    found = await repo.get_many("model-a", ["h1", "missing"])

    # pgvector stores float32.
    assert found.keys() == {"h1"}
    assert found["h1"] == pytest.approx(vector, rel=1e-6)
    assert await repo.get_many("model-b", ["h1"]) == {}


async def test_duplicate_put_is_ignored(session: AsyncSession) -> None:
    repo = EmbeddingCacheRepository(session)
    original = fake_vector("a", EMBEDDING_DIM)
    await repo.put_many("m", {"h": original})

    await repo.put_many("m", {"h": fake_vector("b", EMBEDDING_DIM)})

    assert (await repo.get_many("m", ["h"]))["h"] == pytest.approx(original, rel=1e-6)
