# Technical Design — Buddler

Status: **M1–M6 implemented** (see §10). Guardrails: `AGENTS.md`. Decisions: `docs/adr/`.

Buddler turns a sentence like *"I'm feeling cold"* into an observed, knowledge-backed
decision for a Homematic home: it reads the room, the clock and the weather, matches the
situation against developer-curated **symptom → action** knowledge, lets a bounded agent
check history, and then acts (thermostat) or advises (tea, sweater). Every step is traced.

## 1. Decisions

| Topic | Decision | ADR |
|-------|----------|-----|
| Codebase | Reuse the anomaly-agent plumbing (LLM, MCP, tracing, runs, UI); replace its domain. Migrations squashed into a new baseline. | [0001](adr/0001-repurpose-anomaly-agent.md) |
| Actuation | Agent writes to Homematic **autonomously within bounds**: room allowlist, setpoint range, max delta, writes per run, cooldown, dry-run. | [0002](adr/0002-bounded-autonomous-actuation.md) |
| Knowledge | **Hybrid**: symptom text is embedded (similarity), each action carries structured conditions evaluated deterministically; the LLM chooses only among evaluated candidates. | [0003](adr/0003-hybrid-symptom-action-knowledge.md) |
| Homematic | CCU-Jack VEAP REST API (HTTP/JSON). Rooms → channels mapped in config; `home discover` helps fill it. In-memory fake home for tests/offline. | [0004](adr/0004-ccu-jack-veap-adapter.md) |
| Weather | Open-Meteo forecast API (no key): current temperature, cloud cover, sunshine forecast. Fake for tests/offline. | [0005](adr/0005-open-meteo-weather.md) |
| API auth | Static bearer `API_KEY` (unchanged from the predecessor; JWT is a known gap, §9). | — |

## 2. Domain

| Term | Meaning |
|------|---------|
| Situation | The user's sentence. Untrusted data; summarized by the LLM into a **symptom** phrase plus an optional room mention. |
| Symptom | Curated, embedded phrase ("feeling cold"). Matching is by cosine distance. |
| Action | What to do for a symptom: `advice` (text for the user), `heating_delta` (±°C on the room setpoint), `heating_setpoint` (absolute °C). Has **conditions** and a priority. |
| Conditions | Deterministic checks on observations: local time window, outdoor temperature, sunshine forecast, room temperature, humidity, window state. Each check is `met`, `violated` or `unknown`. |
| Observation | Facts gathered for this run: local time, weather, room climate. |
| Resolution | The run's result: reply to the user, chosen actions with their execution status, advice, citations. |

Refs in LLM context (citations map back to ids; uncited claims are flagged):
`S1…` symptoms, `A1…` candidate actions, `O1` time, `O2` weather, `O3` room climate.

## 3. Workflows (LangGraph)

Both graphs are linear (easy to follow in the trace UI). Nodes reach DB, LLM, Homematic and
weather only through ports injected via `ResolveDeps` / `TrainDeps` (`agents/ports.py`).

### Resolve (user)

`understand → locate → observe → retrieve → evaluate → agent → decide → act → report`

| Node | Kind | Does |
|------|------|------|
| `understand` | LLM | Situation → `{symptom, room_mention, urgency}` (structured). On LLM failure the raw sentence is the symptom (traced). |
| `locate` | det. | Room = `--room` argument › mentioned room (name/alias match) › `DEFAULT_ROOM` › none. |
| `observe` | det. | Local time (or simulated `--at`), weather, room climate. A failing source degrades to "unknown" (traced error), never aborts. |
| `retrieve` | det. | Embed symptom; top-k confirmed symptoms within `RESOLVE_MAX_DISTANCE`; their confirmed actions. |
| `evaluate` | det. | Check every action's conditions against observations; pre-check actuation bounds for device actions. |
| `agent` | LLM loop | Bounded (`AGENT_MAX_STEPS`, `AGENT_MAX_TOOL_CALLS`). Tools: `query` (read-only SQL via MCP on `knowledge.*`, e.g. recent device commands) and `read_room` (climate of another configured room). |
| `decide` | LLM | Chooses candidate actions by ref, writes the reply and advice, cites refs. Only applicable candidates can be executed. |
| `act` | det. | Applies chosen device actions through the actuation guard (ADR 0002); logs every command (`applied`, `dry_run`, `rejected`, `failed`). |
| `report` | det. | Persists the resolution. |

Example (seed knowledge): *"I'm feeling cold"*, living room 19.5 °C, setpoint 20 °C.
At 02:00 the `heating_delta +1` action violates its 06:00–22:00 window, so the agent advises
a warm drink / sweater. At 13:00 under clouds it is applicable: setpoint 20 → 21 °C. At 13:00
with ≥ 2 h sunshine forecast it is violated (`sun_hours_max`), and the "let the sun in" advice
applies instead.

### Train (developer)

`embed → match → propose → review → persist`

| Node | Kind | Does |
|------|------|------|
| `embed` | det. | Embed the symptom phrase. |
| `match` | det. | Nearest existing symptom (any status) within `TRAIN_DUPLICATE_MAX_DISTANCE` and its actions — avoids duplicates and gives the LLM context. |
| `propose` | LLM | Proposes actions with structured conditions from the symptom and an optional developer hint. Device parameters outside bounds are dropped (traced). |
| `review` | human | `interactive` (CLI, LangGraph `interrupt`), `auto` (confirm all), `defer` (keep `proposed`; review later via CLI/API). |
| `persist` | det. | New symptom or attach to the matched one; never overwrites existing entries. |

Human-authored knowledge skips the LLM: `buddler knowledge import FILE.json` stores entries as
`confirmed`, provenance `human`.

## 4. Data Model

Schema `app` (read/write role `app_rw`):

| Table | Columns (main) |
|-------|----------------|
| `symptom` | id, text (unique), embedding `vector(1536)` HNSW, status, provenance, prompt_version, created/updated/reviewed_at |
| `action` | id, symptom_id → symptom (cascade), kind, title, instruction, params JSONB, conditions JSONB, priority, status, provenance, prompt_version, created/reviewed_at |
| `run` | id, kind (`train`/`resolve`), subject (≤ 200 chars), status, started/finished_at, tokens_in/out, error |
| `trace_event` | run_id, seq, node, parent_node, type, summary, payload JSONB, tokens, latency_ms, ts |
| `resolution` | run_id (unique), body JSONB, citations JSONB |
| `device_command` | id, run_id, action_id, room, address, parameter, old_value, new_value, status, reason, created_at |
| `embedding_cache` | (model, text_hash) → embedding |

`params` / `conditions` are validated by Pydantic (`knowledge/conditions.py`) at every boundary.

Schema `knowledge` (views for the read-only MCP role `mcp_ro`; no embeddings, no user text):
`knowledge.symptoms`, `knowledge.actions` (confirmed only), `knowledge.device_commands`.

## 5. Interfaces

### CLI (`buddler`)

| Command | Purpose |
|---------|---------|
| `ask "SENTENCE" [--room R] [--at HH:MM] [--dry-run] [--no-mcp] [--json]` | Resolve workflow |
| `train "SYMPTOM" [--hint TEXT] [--review interactive\|auto\|defer]` | Train workflow |
| `knowledge list [--status S]` / `import FILE` / `confirm ID…` / `reject ID…` | Curate knowledge (ids of symptoms or actions) |
| `home rooms` / `home discover` | Current climate of configured rooms / list CCU-Jack rooms + climate channels |
| `weather` | Current observation for the configured location |
| `runs list` / `runs show RUN_ID` | Runs and their trace tree |
| `graph train\|resolve` | Mermaid source of a workflow |
| `db upgrade`, `llm smoke`, `serve`, `config`, `version` | Operations |

### REST API

Auth: `Authorization: Bearer $API_KEY` on everything except `/health` (static key, constant-time compare; empty key = API locked).

| Method | Path | Notes |
|--------|------|-------|
| GET | `/health` | Public |
| POST | `/runs` | `{"kind":"resolve","situation":…,"room"?,"at"?,"dry_run"?,"use_mcp"?}` or `{"kind":"train","symptom":…,"hint"?,"review_mode":"auto"\|"defer"}` → 202 + `run_id` |
| GET | `/runs`, `/runs/{id}` | List; detail incl. resolution |
| GET | `/runs/{id}/events`, `/runs/{id}/stream` | Trace events; SSE (resumable via `Last-Event-ID`) |
| GET | `/knowledge/symptoms?status=` | Symptoms with actions |
| PATCH | `/knowledge/symptoms/{id}`, `/knowledge/actions/{id}` | `{"status": "confirmed"\|"rejected"\|"proposed"}` |
| GET | `/home/rooms` | Configured rooms with current climate |
| GET | `/graphs/{train\|resolve}` | Mermaid + plain-language step explanations |

## 6. Visualization

Static page, no build step: `GET /ui` (run list), `GET /ui/runs/{id}` (trace view). Page is public (holds no data); every data call uses the bearer key (prompted once, kept in `sessionStorage`).

| Element | Behavior |
|---------|----------|
| Run list | Kind, subject, status, duration, tokens; auto-refresh |
| Workflow | Mermaid graph; nodes colored live (pending/running/paused/done/degraded/failed); click = step explanation + timeline filter |
| Timeline | Trace events streamed via SSE (LLM, MCP, Homematic, weather calls as leaves) |
| Event detail | Summary, metrics, JSON payload |
| Resolution | Reply, observations, executed actions (applied/dry-run/rejected), advice, citations |

- All server data rendered via `textContent`; only Mermaid's SVG (from our own graph definition) uses `innerHTML` (enforced by test).
- CSP: scripts self + jsdelivr, `connect-src 'self'`, `frame-ancestors 'none'`.
- CLI equivalent: `runs show` (Rich tree), `graph train|resolve` (Mermaid source).

## 7. Project Layout

```
src/buddler/
  config.py            # pydantic-settings
  cli/                 # Typer commands
  api/routers/  api/dependencies.py  api/main.py  api/run_manager.py
  services/            # resolve/train/knowledge/home/run services, db_ports (port adapters)
  repositories/  models/  schemas/
  agents/              # graphs/{resolve,train,explain}.py, nodes/, state.py, ports.py, schemas.py
  knowledge/           # conditions + params models, condition evaluation, actuation guard
  home/                # Homematic port, CCU-Jack VEAP client, fake home
  weather/             # weather port, Open-Meteo client, fake
  llm/                 # chat + embeddings clients, fakes, embedding cache
  mcp_client/          # MCP client, tool allowlist, SQL guard
  tracing/             # events, tracer, DB sink, Rich renderer
  ui/static/           # trace page (index.html, app.js, styles.css)
  migrations/          # Alembic (packaged; `buddler db upgrade`)
  prompts/{resolve,train}/*.md  # versioned by content hash
data/knowledge/seed.json        # demo knowledge (human-curated)
tests/{unit,integration}
```

## 8. Config (`.env.example`)

| Var | Example |
|-----|---------|
| `LLM_BASE_URL` | `https://router.requesty.ai/v1` |
| `LLM_API_KEY` | — |
| `LLM_CHAT_MODEL` | `openai/gpt-4o-mini` |
| `LLM_EMBED_MODEL` | `openai/text-embedding-3-small` |
| `EMBED_DIM` | `1536` |
| `DATABASE_URL` | `postgresql+asyncpg://app_rw:...@localhost:5432/buddler` |
| `MCP_URL` | `http://localhost:8001/sse` |
| `AGENT_MAX_STEPS` / `AGENT_MAX_TOOL_CALLS` / `LLM_MAX_TOKENS_PER_CALL` | `8` / `6` / `4000` |
| `HOMEMATIC_PROVIDER` | `ccu_jack` \| `fake` |
| `HOMEMATIC_ENDPOINT` | `https://homematic-raspi:2122` |
| `HOMEMATIC_USERNAME` / `HOMEMATIC_PASSWORD` / `HOMEMATIC_VERIFY_TLS` | CCU-Jack basic auth; self-signed cert → `false` |
| `HOMEMATIC_ROOMS` | `{"living room": {"climate": "000A1B2C3D4E5F:1", "aliases": ["wohnzimmer"]}}` |
| `DEFAULT_ROOM` | `living room` |
| `ACTUATION_ROOMS` | `["living room"]` (empty = no writes) |
| `SETPOINT_MIN_C` / `SETPOINT_MAX_C` / `SETPOINT_MAX_DELTA_C` | `16` / `24` / `2` |
| `ACTUATION_MAX_WRITES_PER_RUN` / `ACTUATION_COOLDOWN_MIN` / `DRY_RUN` | `1` / `30` / `false` |
| `WEATHER_PROVIDER` | `open_meteo` \| `fake` |
| `LOCATION_LATITUDE` / `LOCATION_LONGITUDE` / `TIMEZONE` | `52.52` / `13.405` / `Europe/Berlin` |
| `WEATHER_FORECAST_HOURS` | `6` |

## 9. Assumptions / Open Points

- API auth: static bearer API key (deviates from JWT rule; demonstrator, local only). Confirm or require JWT.
- One climate channel per room (HmIP heating group or thermostat channel exposing `ACTUAL_TEMPERATURE`, `SET_POINT_TEMPERATURE`, optional `HUMIDITY`, `WINDOW_STATE`). Optional separate window contact (`STATE`).
- The user's room is given or mentioned; presence detection (motion sensors) is not used yet.
- Setpoint writes go to `SET_POINT_TEMPERATURE` only; HmIP may revert at the next schedule switch point — acceptable for the demo.
- The user's sentence is stored in `run.subject` and the resolution (local demonstrator); it is not exposed to `mcp_ro`.

## 10. Implementation Plan

| # | Milestone | Done when |
|---|-----------|-----------|
| M1 | Repurpose: rename package, drop file-ingestion domain, new baseline schema | `uv run pytest` green, `db upgrade` creates new schema |
| M2 | Ports + adapters: Homematic (CCU-Jack + fake), weather (Open-Meteo + fake), clock | Unit tests with mocked HTTP transports |
| M3 | Knowledge: symptom/action model, conditions, import, CLI/API review | Seed import + confirm/reject via CLI and API |
| M4 | Resolve workflow incl. agent tools and bounded actuation | Scenario tests a) 02:00 → advice, b) 13:00 cloudy → setpoint +1 |
| M5 | Train workflow with interactive/auto/defer review | Unit tests with fake LLM; CLI review |
| M6 | Interfaces: CLI, REST, trace UI, demo script | `scripts/demo.sh` runs end to end with fake home |
| M7 | Real CCU validation | `home discover` / `ask` against the user's CCU-Jack (manual) |

Each milestone = one Conventional Commit series; ask before deviating from this design.
