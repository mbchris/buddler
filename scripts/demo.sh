#!/usr/bin/env bash
# End-to-end demo from a fresh clone: setup -> knowledge -> scenarios -> UI.
#
#   scripts/demo.sh [--train] [--serve]
#
#   --train     also let the AI propose actions for a new symptom (needs LLM_API_KEY)
#   --serve     start the API + trace UI at the end (Ctrl+C to stop)
#
# Runs against the simulated home and weather unless .env says otherwise.
# Env: LLM_API_KEY (used when .env is first created; without it the offline fake LLM runs),
#      DB_PORT / MCP_PORT (host ports for a new .env, default 5432 / 8001).
set -euo pipefail

cd "$(dirname "$0")/.."

TRAIN=false SERVE=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --train) TRAIN=true ;;
    --serve) SERVE=true ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

BD=(uv run buddler)

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
env_value() { grep -E "^$1=" .env | head -1 | cut -d= -f2-; }

step "1/7 Preflight"
command -v uv >/dev/null || { echo "uv missing: https://docs.astral.sh/uv/" >&2; exit 1; }
command -v docker >/dev/null || { echo "docker missing" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "Docker daemon not running" >&2; exit 1; }

step "2/7 Configuration (.env)"
uv run --quiet python scripts/bootstrap_env.py --db-port "${DB_PORT:-5432}" --mcp-port "${MCP_PORT:-8001}"

step "3/7 PostgreSQL + Postgres MCP (docker compose)"
docker compose up -d db mcp
for _ in $(seq 1 60); do
  [[ "$(docker compose ps db --format '{{.Health}}')" == "healthy" ]] && break
  sleep 1
done
[[ "$(docker compose ps db --format '{{.Health}}')" == "healthy" ]] || { echo "db not healthy" >&2; exit 1; }

step "4/7 Dependencies + schema + knowledge"
uv sync --quiet
"${BD[@]}" db upgrade
"${BD[@]}" knowledge import data/knowledge/seed.json
"${BD[@]}" knowledge list
"${BD[@]}" home rooms

if $TRAIN; then
  step "4b Train: AI proposes actions for a new symptom (deferred review)"
  "${BD[@]}" train "room too dark" --hint "advice only, no devices" --review defer
fi

step "5/7 Scenario a) 02:00 — no heating at night, advice instead"
"${BD[@]}" ask "I'm feeling cold" --at 02:00

step "6/7 Scenario b) 13:00, cloudy — heating +1 °C (living room)"
"${BD[@]}" ask "Mir ist kalt im Wohnzimmer" --at 13:00

step "7/7 Observe"
"${BD[@]}" runs list --limit 5
API_PORT="$(env_value API_PORT)"; API_PORT="${API_PORT:-8000}"
cat <<EOF

Trace tree:   uv run buddler runs show <run_id>
Graphs:       uv run buddler graph resolve
Trace UI:     http://127.0.0.1:${API_PORT}/ui   (API key: grep API_KEY .env)
API docs:     http://127.0.0.1:${API_PORT}/docs
EOF

if $SERVE; then
  step "Serving API + trace UI on port ${API_PORT} (Ctrl+C to stop)"
  exec "${BD[@]}" serve --port "$API_PORT"
fi
