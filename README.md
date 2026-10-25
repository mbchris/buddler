# buddler

Agentic AI demonstrator: Connects to the Homematic home automation system of a user. Based on the users queries, it acts individually based on the situation it finds.

Knowledge of how to act and user preferences are stored on a Postgres server connected using MCP. 

Every step is traced and visible live.

| Doc | Content |
|-----|---------|
| [AGENTS.md](AGENTS.md) | Guardrails for humans and coding agents |
| [docs/design.md](docs/design.md) | Technical design, data model, interfaces |
| [docs/adr/](docs/adr/README.md) | Architecture decisions |

## Quickstart (demo)

Prerequisites: Docker (running), [uv](https://docs.astral.sh/uv/), Python ≥ 3.12 (uv installs it).

```bash
LLM_API_KEY=<requesty key> scripts/demo.sh --serve
```

| Step | What happens |
|------|--------------|
| Configuration | Creates `.env` from `.env.example` with generated DB/API secrets; simulated home + weather by default |
| Infrastructure | Starts PostgreSQL (pgvector) and the read-only Postgres MCP server via Docker Compose |
| Schema + knowledge | `db upgrade`, imports [data/knowledge/seed.json](data/knowledge/seed.json) (feeling cold / warm, stuffy air) |
| Scenario a) | `ask "I'm feeling cold" --at 02:00` — heating action violates its time window → tea / sweater advice |
| Scenario b) | `ask "Mir ist kalt im Wohnzimmer" --at 13:00` — cloudy, room below comfort → living room setpoint 20 → 21 °C |
| Observe | Lists runs; `--serve` starts the API and trace UI |

Without `LLM_API_KEY` the offline fake LLM runs; the pipeline completes but its placeholder answers match no knowledge.

## Trace UI

`buddler serve`, then open `http://127.0.0.1:8000/ui` and enter `API_KEY` from `.env`.
The run list shows every train/resolve run; a run page shows the workflow graph with live node
states, the streamed timeline (LLM, MCP, Homematic and weather calls), event payloads, a
plain-language explanation of each step, and the resolution: reply, observations, executed
actions (applied / dry run / rejected) and advice with citations.

## Architecture

```mermaid
flowchart LR
  user([User sentence]) --> cli[CLI / REST API]
  cli --> svc[Services]
  svc --> graph[LangGraph resolve / train]
  graph -->|ports| llm[llm/ Requesty]
  graph -->|ports| home[home/ CCU-Jack VEAP]
  graph -->|ports| weather[weather/ Open-Meteo]
  graph -->|ports| repo[repositories/ pgvector]
  graph -->|execute_sql| mcp[Postgres MCP, read-only]
  repo --> db[(PostgreSQL)]
  mcp --> db
  graph --> trace[tracing/] --> db
  ui[Trace UI] -->|REST + SSE| cli
```

Resolve: `understand → locate → observe → retrieve → evaluate → agent → decide → act → report`.
Device writes pass a deterministic actuation guard ([ADR 0002](docs/adr/0002-bounded-autonomous-actuation.md)).

## Manual setup

| Step | Command |
|------|---------|
| Config | `uv run python scripts/bootstrap_env.py` (then edit `.env`: `LLM_API_KEY`, Homematic) |
| Database + MCP | `docker compose up -d db mcp` |
| Dependencies | `uv sync` |
| Schema | `uv run buddler db upgrade` |
| Knowledge | `uv run buddler knowledge import data/knowledge/seed.json` |
| Real CCU | `HOMEMATIC_PROVIDER=ccu_jack`, endpoint/credentials, then `uv run buddler home discover` → `HOMEMATIC_ROOMS`, `ACTUATION_ROOMS` |
| Real weather | `WEATHER_PROVIDER=open_meteo`, `LOCATION_LATITUDE` / `LOCATION_LONGITUDE` |
| Checks | `uv run ruff check && uv run ruff format --check && uv run mypy src tests && uv run pytest` |
| Integration tests | `uv run pytest -m integration` (needs the Docker DB) |

## Usage

| Task | CLI | API |
|------|-----|-----|
| Resolve a situation | `buddler ask "I'm feeling cold" [--room R] [--at HH:MM] [--dry-run]` | `POST /runs {"kind":"resolve","situation":"…"}` |
| Propose knowledge | `buddler train "feeling cold" [--hint …] [--review auto\|defer]` | `POST /runs {"kind":"train","symptom":"…"}` |
| Import knowledge | `buddler knowledge import FILE.json` | — |
| Review knowledge | `buddler knowledge list` / `confirm ID…` / `reject ID…` | `GET /knowledge/symptoms`, `PATCH /knowledge/{symptoms\|actions}/{id}` |
| Home / weather | `buddler home rooms`, `buddler home discover`, `buddler weather` | `GET /home/rooms` |
| Runs and traces | `buddler runs list`, `buddler runs show RUN_ID` | `GET /runs`, `/runs/{id}`, `/runs/{id}/events`, `/runs/{id}/stream` |
| Workflow graphs | `buddler graph resolve\|train` | `GET /graphs/{resolve\|train}` |
