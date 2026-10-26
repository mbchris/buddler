# 0002 — Bounded autonomous actuation

## Status
Accepted (2026-10-25)

## Context
The demo should show the agent changing the home (e.g. raising a thermostat), not just advising.
LLM output must never directly control devices.

## Decision
- The LLM only selects candidate actions by ref. Device parameters come from curated knowledge.
- A deterministic guard (`knowledge/actuation.py`) runs before every write and rejects when:
  the action's conditions are not all met; the room is not in `ACTUATION_ROOMS`; the resulting
  setpoint is outside `SETPOINT_MIN_C..SETPOINT_MAX_C`; the change exceeds `SETPOINT_MAX_DELTA_C`;
  `ACTUATION_MAX_WRITES_PER_RUN` is reached; or the room was written within
  `ACTUATION_COOLDOWN_MIN` minutes.
- `DRY_RUN` / `--dry-run` logs the command without writing.
- Every command (applied, dry-run, rejected, failed) is stored in `device_command` and traced.

## Consequences
- With the default empty `ACTUATION_ROOMS` nothing is written; enabling writes is an explicit
  config step.
- Conditions with `unknown` outcome (sensor missing) block device actions but not advice.
