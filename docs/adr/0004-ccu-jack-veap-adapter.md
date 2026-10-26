# 0004 — Homematic via CCU-Jack VEAP

## Status
Accepted (2026-10-25)

## Context
CCU-Jack exposes the CCU's devices via a REST API (VEAP: `/device/{address}/{channel}/{parameter}/~pv`,
JSON `{"v": value, "ts": …, "s": …}`) and MQTT. A request/response API fits the workflow.

## Decision
- `home/ccu_jack.py` reads and writes process values over HTTPS (basic auth, optional TLS
  verification) using `httpx2` (already used by the MCP client).
- Rooms are mapped to channels in config (`HOMEMATIC_ROOMS`), with aliases for matching user
  wording. `buddler home discover` lists CCU rooms and channels with `SET_POINT_TEMPERATURE` to
  help fill the map.
- `home/fake.py` simulates a small home for tests and offline demos (`HOMEMATIC_PROVIDER=fake`).
- The agent never gets a generic HTTP tool; it only gets `read_room` for configured rooms.

## Consequences
- No automatic room discovery at runtime: predictable, but config must be kept in sync with the CCU.
- MQTT (push updates) is not used.
