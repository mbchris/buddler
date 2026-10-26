"""Actuation guard (ADR 0002): the deterministic gate in front of every device write."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from buddler.config import Settings
from buddler.knowledge.conditions import ActionParams, CheckOutcome, ConditionCheck
from buddler.models.enums import ActionKind

# HmIP thermostats accept 0.5 °C steps.
SETPOINT_STEP_C = 0.5


@dataclass(frozen=True)
class ActuationBounds:
    rooms: frozenset[str]
    setpoint_min_c: float
    setpoint_max_c: float
    max_delta_c: float
    max_writes_per_run: int
    cooldown: timedelta

    @classmethod
    def from_settings(cls, settings: Settings) -> "ActuationBounds":
        return cls(
            rooms=frozenset(settings.actuation_rooms),
            setpoint_min_c=settings.setpoint_min_c,
            setpoint_max_c=settings.setpoint_max_c,
            max_delta_c=settings.setpoint_max_delta_c,
            max_writes_per_run=settings.actuation_max_writes_per_run,
            cooldown=timedelta(minutes=settings.actuation_cooldown_min),
        )


@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    reason: str
    target_c: float | None = None


def round_setpoint(value: float) -> float:
    return round(value / SETPOINT_STEP_C) * SETPOINT_STEP_C


def target_setpoint(
    kind: ActionKind, params: ActionParams, current_c: float | None
) -> float | None:
    if kind is ActionKind.HEATING_SETPOINT and params.setpoint_c is not None:
        return round_setpoint(params.setpoint_c)
    if kind is ActionKind.HEATING_DELTA and params.delta_c is not None and current_c is not None:
        return round_setpoint(current_c + params.delta_c)
    return None


def check_static(
    kind: ActionKind,
    params: ActionParams,
    *,
    room: str | None,
    current_c: float | None,
    checks: Sequence[ConditionCheck],
    bounds: ActuationBounds,
) -> GuardDecision:
    """Everything decidable before the run starts writing (shown to the LLM as `executable`)."""
    violated = [c.condition for c in checks if c.outcome is CheckOutcome.VIOLATED]
    if violated:
        return GuardDecision(False, f"conditions violated: {', '.join(violated)}")
    unknown = [c.condition for c in checks if c.outcome is CheckOutcome.UNKNOWN]
    if unknown:
        return GuardDecision(False, f"conditions unknown: {', '.join(unknown)}")
    if room is None:
        return GuardDecision(False, "no room identified")
    if room not in bounds.rooms:
        return GuardDecision(False, f"room {room!r} not in ACTUATION_ROOMS")
    if current_c is None:
        return GuardDecision(False, "current setpoint unknown")
    target = target_setpoint(kind, params, current_c)
    if target is None:
        return GuardDecision(False, f"{kind} has no device target")
    if not bounds.setpoint_min_c <= target <= bounds.setpoint_max_c:
        return GuardDecision(
            False,
            f"target {target:g} C outside {bounds.setpoint_min_c:g}-{bounds.setpoint_max_c:g} C",
        )
    if abs(target - current_c) > bounds.max_delta_c:
        return GuardDecision(
            False, f"change {target - current_c:+g} C exceeds max delta {bounds.max_delta_c:g} C"
        )
    if target == current_c:
        return GuardDecision(False, f"setpoint already {target:g} C")
    return GuardDecision(True, f"setpoint {current_c:g} -> {target:g} C", target)


def check_runtime(
    static: GuardDecision,
    *,
    writes_this_run: int,
    last_write_at: datetime | None,
    now: datetime,
    bounds: ActuationBounds,
) -> GuardDecision:
    """Limits that depend on what already happened (this run, earlier runs)."""
    if not static.allowed:
        return static
    if writes_this_run >= bounds.max_writes_per_run:
        return GuardDecision(False, f"max {bounds.max_writes_per_run} device write(s) per run")
    if last_write_at is not None and now - last_write_at < bounds.cooldown:
        minutes = int((now - last_write_at).total_seconds() // 60)
        return GuardDecision(
            False,
            f"room written {minutes} min ago "
            f"(cooldown {int(bounds.cooldown.total_seconds() // 60)} min)",
        )
    return static
