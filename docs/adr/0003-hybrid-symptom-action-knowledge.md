# 0003 — Hybrid symptom → action knowledge

## Status
Accepted (2026-10-25)

## Context
Users phrase situations freely ("I'm freezing", "es ist kalt"), so matching needs semantic
similarity. Whether an action is appropriate depends on hard facts (time, weather, temperature)
that should be checked reproducibly, not guessed by the LLM.

## Decision
- `symptom`: curated phrase with an embedding (pgvector, cosine). Retrieval is a deterministic
  repository query.
- `action`: belongs to a symptom; `kind`, `params`, `priority` and structured `conditions`
  (time window, outdoor/room temperature, sunshine forecast, humidity, window state).
- The `evaluate` node checks conditions deterministically (`met` / `violated` / `unknown`).
  The LLM decides among candidates and writes the reply; it cannot add device actions.
- Only `confirmed` symptoms and actions are used. Human entries are imported as confirmed;
  LLM proposals need review. Existing entries are never overwritten automatically.

## Consequences
- Condition vocabulary is a code contract (Pydantic); new condition types need code + prompt
  changes.
- Changing the embedding model requires a migration and re-embedding of symptoms.
