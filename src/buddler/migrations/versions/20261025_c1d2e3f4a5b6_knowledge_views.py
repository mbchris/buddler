"""Knowledge schema: read-only views for the Postgres MCP server (AGENTS.md §5).

Views expose confirmed knowledge and the device command log only: no embeddings, no user
sentences, no trace payloads. Views run with the owner's (app_rw) rights, so mcp_ro needs
no access to the `app` schema.

Revision ID: c1d2e3f4a5b6
Revises: b0d1e2f3a4b5
Create Date: 2026-10-25 22:05:00

"""

from collections.abc import Sequence

from alembic import op

revision: str = "c1d2e3f4a5b6"
down_revision: str | Sequence[str] | None = "b0d1e2f3a4b5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

MCP_ROLE = "mcp_ro"


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS knowledge")
    op.execute("REVOKE ALL ON SCHEMA knowledge FROM PUBLIC")
    op.execute(
        """
        CREATE VIEW knowledge.symptoms AS
        SELECT id, text, provenance, created_at, reviewed_at
        FROM app.symptom
        WHERE status = 'confirmed'
        """
    )
    op.execute(
        """
        CREATE VIEW knowledge.actions AS
        SELECT a.id, a.symptom_id, s.text AS symptom, a.kind, a.title, a.instruction,
               a.params, a.conditions, a.priority, a.provenance, a.created_at
        FROM app.action a
        JOIN app.symptom s ON s.id = a.symptom_id
        WHERE a.status = 'confirmed' AND s.status = 'confirmed'
        """
    )
    op.execute(
        """
        CREATE VIEW knowledge.device_commands AS
        SELECT id, run_id, action_id, room, parameter, old_value, new_value, status, reason,
               created_at
        FROM app.device_command
        """
    )
    op.execute("COMMENT ON VIEW knowledge.symptoms IS 'Developer-confirmed symptoms.'")
    op.execute(
        "COMMENT ON VIEW knowledge.actions IS "
        "'Confirmed actions per symptom with structured params and conditions.'"
    )
    op.execute(
        "COMMENT ON VIEW knowledge.device_commands IS "
        "'Every device action chosen by the agent: applied, dry_run, rejected or failed.'"
    )
    # Role is created by docker/postgres/init; skip grants where it does not exist.
    op.execute(
        f"""
        DO $$
        BEGIN
          IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{MCP_ROLE}') THEN
            GRANT USAGE ON SCHEMA knowledge TO {MCP_ROLE};
            GRANT SELECT ON ALL TABLES IN SCHEMA knowledge TO {MCP_ROLE};
            ALTER DEFAULT PRIVILEGES IN SCHEMA knowledge GRANT SELECT ON TABLES TO {MCP_ROLE};
          END IF;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS knowledge.device_commands")
    op.execute("DROP VIEW IF EXISTS knowledge.actions")
    op.execute("DROP VIEW IF EXISTS knowledge.symptoms")
    op.execute("DROP SCHEMA IF EXISTS knowledge")
