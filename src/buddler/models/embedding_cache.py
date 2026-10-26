"""Cross-file embedding cache keyed by (model, text hash) (design §2)."""

from pgvector.sqlalchemy import Vector
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from buddler.models.base import EMBEDDING_DIM, Base, CreatedAtMixin

MODEL_NAME_LENGTH = 128


class EmbeddingCacheEntry(CreatedAtMixin, Base):
    __tablename__ = "embedding_cache"

    # Vectors from different models are not comparable; the model is part of the key.
    model: Mapped[str] = mapped_column(String(MODEL_NAME_LENGTH), primary_key=True)
    text_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
