"""Workflow runs, their trace events, results and device commands."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from buddler.models.base import Base, CreatedAtMixin, UuidPkMixin, str_enum
from buddler.models.enums import CommandStatus, RunKind, RunStatus, TraceEventType

# JSON payloads are schemaless by nature; validated by Pydantic at the service boundary.
JsonObject = dict[str, Any]
SUBJECT_LENGTH = 200


class Run(UuidPkMixin, Base):
    __tablename__ = "run"

    kind: Mapped[RunKind] = mapped_column(str_enum(RunKind))
    # The situation (resolve) or symptom (train), truncated; shown in run lists.
    subject: Mapped[str] = mapped_column(String(SUBJECT_LENGTH))
    status: Mapped[RunStatus] = mapped_column(str_enum(RunStatus), default=RunStatus.PENDING)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)

    events: Mapped[list["TraceEvent"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="TraceEvent.seq"
    )


class TraceEvent(Base):
    __tablename__ = "trace_event"
    __table_args__ = (UniqueConstraint("run_id", "seq"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("run.id", ondelete="CASCADE"))
    seq: Mapped[int] = mapped_column(Integer)
    node: Mapped[str] = mapped_column(String(64))
    parent_node: Mapped[str | None] = mapped_column(String(64))
    type: Mapped[TraceEventType] = mapped_column(str_enum(TraceEventType))
    summary: Mapped[str] = mapped_column(Text)
    payload: Mapped[JsonObject] = mapped_column(JSONB, default=dict)
    tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    run: Mapped[Run] = relationship(back_populates="events")


class Resolution(UuidPkMixin, CreatedAtMixin, Base):
    __tablename__ = "resolution"

    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("run.id", ondelete="CASCADE"), unique=True)
    body: Mapped[JsonObject] = mapped_column(JSONB)
    citations: Mapped[list[JsonObject]] = mapped_column(JSONB, default=list)


class DeviceCommand(UuidPkMixin, CreatedAtMixin, Base):
    """Audit log of every device action the agent chose (ADR 0002), executed or not."""

    __tablename__ = "device_command"
    __table_args__ = (Index("ix_device_command_room_status", "room", "status", "created_at"),)

    run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("run.id", ondelete="SET NULL"))
    action_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("action.id", ondelete="SET NULL")
    )
    room: Mapped[str | None] = mapped_column(String(64))
    parameter: Mapped[str] = mapped_column(String(64))
    old_value: Mapped[float | None] = mapped_column(Float)
    new_value: Mapped[float | None] = mapped_column(Float)
    status: Mapped[CommandStatus] = mapped_column(str_enum(CommandStatus))
    reason: Mapped[str] = mapped_column(Text)
