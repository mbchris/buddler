# 0001 — Repurpose the anomaly-agent codebase

## Status
Accepted (2026-10-25)

## Context
The code came from an agentic log/medical anomaly demonstrator. Its infrastructure (LLM gateway
wrapper, Postgres MCP client with SQL guard, tracing, run manager, REST auth, trace UI, Alembic)
fits Buddler. Its domain (streaming file ingestion, signatures, profiles, redaction, anomalies)
does not. No Buddler database exists yet.

## Decision
- Keep the plumbing unchanged where possible; rename the package to `buddler`.
- Delete the file-ingestion domain (`ingestion/`, `profiles/`, `redaction/`, uploads, anomaly
  model/repository/routes, old prompts and graphs).
- Replace the old migration chain with a new baseline revision (no deployed database to migrate).

## Consequences
- AGENTS.md §3 (large input handling) no longer applies and is replaced by external-system rules.
- Old databases from the predecessor are not upgradable; recreate the Docker volume.
