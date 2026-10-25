"""Baseline schema for Buddler (ADR 0001): knowledge, runs, traces, resolutions, commands.

Replaces the predecessor's migration chain; there was no Buddler database to migrate.

Revision ID: b0d1e2f3a4b5
Revises:
Create Date: 2026-10-25 22:00:00

"""

from collections.abc import Sequence
from datetime import datetime

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "b0d1e2f3a4b5"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "app"
EMBEDDING_DIM = 1536

KNOWLEDGE_STATUS = ("proposed", "confirmed", "rejected")
PROVENANCE = ("human", "llm")
ACTION_KIND = ("advice", "heating_delta", "heating_setpoint")
RUN_KIND = ("train", "resolve")
RUN_STATUS = ("pending", "running", "waiting_review", "succeeded", "failed")
COMMAND_STATUS = ("applied", "dry_run", "rejected", "failed")
TRACE_EVENT_TYPE = (
    "node_start",
    "node_end",
    "llm_call",
    "embedding_call",
    "tool_call",
    "mcp_call",
    "home_call",
    "weather_call",
    "error",
)


def _enum(
    column: str, values: Sequence[str], table: str, enum_name: str
) -> tuple[sa.Column[str], sa.CheckConstraint]:
    """VARCHAR + CHECK, matching models.base.str_enum."""
    listed = ", ".join(f"'{v}'" for v in values)
    return (
        sa.Column(column, sa.String(length=32), nullable=False),
        sa.CheckConstraint(f"{column} IN ({listed})", name=op.f(f"ck_{table}_{enum_name}")),
    )


def _created_at() -> sa.Column[datetime]:
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
    )


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    op.execute(f"REVOKE ALL ON SCHEMA {SCHEMA} FROM PUBLIC")

    op.create_table(
        "symptom",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("text", sa.String(length=255), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(dim=EMBEDDING_DIM), nullable=True),
        *_enum("status", KNOWLEDGE_STATUS, "symptom", "knowledgestatus"),
        *_enum("provenance", PROVENANCE, "symptom", "provenance"),
        sa.Column("prompt_version", sa.String(length=64), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_symptom")),
        sa.UniqueConstraint("text", name=op.f("uq_symptom_text")),
        schema=SCHEMA,
    )
    op.create_index("ix_symptom_status", "symptom", ["status"], schema=SCHEMA)
    op.create_index(
        "ix_symptom_embedding_hnsw",
        "symptom",
        ["embedding"],
        schema=SCHEMA,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    op.create_table(
        "action",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("symptom_id", sa.Uuid(), nullable=False),
        *_enum("kind", ACTION_KIND, "action", "actionkind"),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("instruction", sa.Text(), nullable=False),
        sa.Column("params", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("conditions", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        *_enum("status", KNOWLEDGE_STATUS, "action", "knowledgestatus"),
        *_enum("provenance", PROVENANCE, "action", "provenance"),
        sa.Column("prompt_version", sa.String(length=64), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        _created_at(),
        sa.ForeignKeyConstraint(
            ["symptom_id"],
            [f"{SCHEMA}.symptom.id"],
            name=op.f("fk_action_symptom_id_symptom"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_action")),
        schema=SCHEMA,
    )
    op.create_index("ix_action_symptom_status", "action", ["symptom_id", "status"], schema=SCHEMA)

    op.create_table(
        "run",
        sa.Column("id", sa.Uuid(), nullable=False),
        *_enum("kind", RUN_KIND, "run", "runkind"),
        sa.Column("subject", sa.String(length=200), nullable=False),
        *_enum("status", RUN_STATUS, "run", "runstatus"),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("tokens_in", sa.Integer(), nullable=False),
        sa.Column("tokens_out", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_run")),
        schema=SCHEMA,
    )

    op.create_table(
        "trace_event",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("node", sa.String(length=64), nullable=False),
        sa.Column("parent_node", sa.String(length=64), nullable=True),
        *_enum("type", TRACE_EVENT_TYPE, "trace_event", "traceeventtype"),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("tokens", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column(
            "ts", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            [f"{SCHEMA}.run.id"],
            name=op.f("fk_trace_event_run_id_run"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_trace_event")),
        sa.UniqueConstraint("run_id", "seq", name=op.f("uq_trace_event_run_id")),
        schema=SCHEMA,
    )

    op.create_table(
        "resolution",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("body", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("citations", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        _created_at(),
        sa.ForeignKeyConstraint(
            ["run_id"],
            [f"{SCHEMA}.run.id"],
            name=op.f("fk_resolution_run_id_run"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_resolution")),
        sa.UniqueConstraint("run_id", name=op.f("uq_resolution_run_id")),
        schema=SCHEMA,
    )

    op.create_table(
        "device_command",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=True),
        sa.Column("action_id", sa.Uuid(), nullable=True),
        sa.Column("room", sa.String(length=64), nullable=True),
        sa.Column("parameter", sa.String(length=64), nullable=False),
        sa.Column("old_value", sa.Float(), nullable=True),
        sa.Column("new_value", sa.Float(), nullable=True),
        *_enum("status", COMMAND_STATUS, "device_command", "commandstatus"),
        sa.Column("reason", sa.Text(), nullable=False),
        _created_at(),
        sa.ForeignKeyConstraint(
            ["run_id"],
            [f"{SCHEMA}.run.id"],
            name=op.f("fk_device_command_run_id_run"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["action_id"],
            [f"{SCHEMA}.action.id"],
            name=op.f("fk_device_command_action_id_action"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_device_command")),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_device_command_room_status",
        "device_command",
        ["room", "status", "created_at"],
        schema=SCHEMA,
    )

    op.create_table(
        "embedding_cache",
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("text_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(dim=EMBEDDING_DIM), nullable=False),
        _created_at(),
        sa.PrimaryKeyConstraint("model", "text_hash", name=op.f("pk_embedding_cache")),
        schema=SCHEMA,
    )


def downgrade() -> None:
    for table in (
        "embedding_cache",
        "device_command",
        "resolution",
        "trace_event",
        "run",
        "action",
        "symptom",
    ):
        op.drop_table(table, schema=SCHEMA)
