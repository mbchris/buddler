"""Curated symptom -> action knowledge (AGENTS.md §4, ADR 0003)."""

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from buddler.models.base import EMBEDDING_DIM, Base, CreatedAtMixin, UuidPkMixin, str_enum
from buddler.models.enums import ActionKind, KnowledgeStatus, Provenance

SYMPTOM_TEXT_LENGTH = 255


class Symptom(UuidPkMixin, CreatedAtMixin, Base):
    __tablename__ = "symptom"
    __table_args__ = (
        Index("ix_symptom_status", "status"),
        Index(
            "ix_symptom_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    text: Mapped[str] = mapped_column(String(SYMPTOM_TEXT_LENGTH), unique=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    status: Mapped[KnowledgeStatus] = mapped_column(
        str_enum(KnowledgeStatus), default=KnowledgeStatus.PROPOSED
    )
    provenance: Mapped[Provenance] = mapped_column(str_enum(Provenance))
    prompt_version: Mapped[str | None] = mapped_column(String(64))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    actions: Mapped[list["Action"]] = relationship(
        back_populates="symptom",
        cascade="all, delete-orphan",
        order_by="[Action.priority, Action.created_at]",
    )


class Action(UuidPkMixin, CreatedAtMixin, Base):
    __tablename__ = "action"
    __table_args__ = (Index("ix_action_symptom_status", "symptom_id", "status"),)

    symptom_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("symptom.id", ondelete="CASCADE"))
    kind: Mapped[ActionKind] = mapped_column(str_enum(ActionKind))
    title: Mapped[str] = mapped_column(String(255))
    # Advice text for the user, or what the device action achieves.
    instruction: Mapped[str] = mapped_column(Text)
    # Validated by knowledge.conditions.ActionParams / ActionConditions.
    params: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    conditions: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    # Lower = preferred.
    priority: Mapped[int] = mapped_column(Integer, default=100)
    status: Mapped[KnowledgeStatus] = mapped_column(
        str_enum(KnowledgeStatus), default=KnowledgeStatus.PROPOSED
    )
    provenance: Mapped[Provenance] = mapped_column(str_enum(Provenance))
    prompt_version: Mapped[str | None] = mapped_column(String(64))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    symptom: Mapped[Symptom] = relationship(back_populates="actions")
