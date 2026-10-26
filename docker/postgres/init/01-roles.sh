#!/usr/bin/env bash
# Runs once on first DB init. Table/view grants are applied by Alembic.
set -euo pipefail

psql -v ON_ERROR_STOP=1 \
  --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  -v app_rw_password="$APP_RW_PASSWORD" \
  -v mcp_ro_password="$MCP_RO_PASSWORD" <<'SQL'
CREATE ROLE app_rw LOGIN PASSWORD :'app_rw_password';
CREATE ROLE mcp_ro LOGIN PASSWORD :'mcp_ro_password';
-- mcp_ro must never write, even if a grant is added by mistake.
ALTER ROLE mcp_ro SET default_transaction_read_only = on;
SQL

setup_database() {
  local db_name="$1"
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$db_name" -v db_name="$db_name" <<'SQL'
CREATE EXTENSION IF NOT EXISTS vector;
ALTER DATABASE :"db_name" OWNER TO app_rw;
ALTER SCHEMA public OWNER TO app_rw;
GRANT CONNECT ON DATABASE :"db_name" TO mcp_ro;
SQL
}

setup_database "$POSTGRES_DB"

# Integration tests run against a separate database (AGENTS.md §9).
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  -c "CREATE DATABASE ${POSTGRES_DB}_test"
setup_database "${POSTGRES_DB}_test"
