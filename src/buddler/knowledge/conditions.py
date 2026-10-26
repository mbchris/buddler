"""Action conditions and parameters (ADR 0003) and their deterministic evaluation.

These models are a contract shared by the DB (JSONB), the LLM (train proposals), the seed
file format and the trace UI. Unknown fields are rejected, so neither a curator typo nor an
LLM can introduce a condition that is silently ignored.
"""

from datetime import datetime, time
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from buddler.home.ports import RoomClimate
from buddler.models.enums import ActionKind
from buddler.weather.ports import WeatherObservation


class ActionConditions(BaseModel):
    """All set conditions must hold; unset ones are ignored."""

    model_config = ConfigDict(extra="forbid")

    time_from: time | None = Field(default=None, description="Local time window start (HH:MM).")
    time_to: time | None = Field(
        default=None, description="Window end, exclusive; wraps midnight if before time_from."
    )
    outdoor_temp_min_c: float | None = None
    outdoor_temp_max_c: float | None = None
    sun_hours_min: float | None = Field(default=None, description="Forecast sunshine, hours.")
    sun_hours_max: float | None = None
    room_temp_min_c: float | None = None
    room_temp_max_c: float | None = None
    humidity_min_pct: float | None = None
    humidity_max_pct: float | None = None
    window_open: bool | None = None

    @model_validator(mode="after")
    def _window_complete(self) -> Self:
        if (self.time_from is None) != (self.time_to is None):
            raise ValueError("time_from and time_to must be set together")
        return self


class ActionParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delta_c: float | None = Field(default=None, description="heating_delta: +/- degrees C.")
    setpoint_c: float | None = Field(default=None, description="heating_setpoint: degrees C.")


def validate_params(kind: ActionKind, params: ActionParams) -> ActionParams:
    """Raise ValueError when params do not fit the action kind."""
    if kind is ActionKind.HEATING_DELTA and (params.delta_c is None or params.delta_c == 0):
        raise ValueError("heating_delta needs a non-zero delta_c")
    if kind is ActionKind.HEATING_SETPOINT and params.setpoint_c is None:
        raise ValueError("heating_setpoint needs setpoint_c")
    if kind is ActionKind.ADVICE and (params.delta_c is not None or params.setpoint_c is not None):
        raise ValueError("advice takes no device params")
    return params


class Observations(BaseModel):
    local_time: datetime
    # True when `local_time` was given (`--at`) instead of read from the clock.
    simulated_time: bool = False
    weather: WeatherObservation | None
    climate: RoomClimate | None


class CheckOutcome(StrEnum):
    MET = "met"
    VIOLATED = "violated"
    # Observation missing (sensor absent, weather unreachable).
    UNKNOWN = "unknown"


class ConditionCheck(BaseModel):
    condition: str
    expected: str
    observed: str | None
    outcome: CheckOutcome


def in_window(now: time, start: time, end: time) -> bool:
    if start <= end:
        return start <= now < end
    return now >= start or now < end


def _bound_check(
    name: str, observed: float | None, minimum: float | None, maximum: float | None, unit: str
) -> ConditionCheck | None:
    if minimum is None and maximum is None:
        return None
    expected = " ".join(
        part
        for part in (
            f">= {minimum:g}{unit}" if minimum is not None else "",
            f"<= {maximum:g}{unit}" if maximum is not None else "",
        )
        if part
    )
    if observed is None:
        return ConditionCheck(
            condition=name, expected=expected, observed=None, outcome=CheckOutcome.UNKNOWN
        )
    ok = (minimum is None or observed >= minimum) and (maximum is None or observed <= maximum)
    return ConditionCheck(
        condition=name,
        expected=expected,
        observed=f"{observed:g}{unit}",
        outcome=CheckOutcome.MET if ok else CheckOutcome.VIOLATED,
    )


def evaluate_conditions(
    conditions: ActionConditions, observations: Observations
) -> list[ConditionCheck]:
    checks: list[ConditionCheck] = []
    if conditions.time_from is not None and conditions.time_to is not None:
        now = observations.local_time.time().replace(second=0, microsecond=0)
        start, end = conditions.time_from, conditions.time_to
        checks.append(
            ConditionCheck(
                condition="time_window",
                expected=f"{start:%H:%M}-{end:%H:%M}",
                observed=f"{now:%H:%M}",
                outcome=CheckOutcome.MET if in_window(now, start, end) else CheckOutcome.VIOLATED,
            )
        )
    weather = observations.weather
    climate = observations.climate
    bounds = (
        (
            "outdoor_temp",
            weather.outdoor_temp_c if weather else None,
            conditions.outdoor_temp_min_c,
            conditions.outdoor_temp_max_c,
            " C",
        ),
        (
            "sun_hours",
            weather.sun_hours_forecast if weather else None,
            conditions.sun_hours_min,
            conditions.sun_hours_max,
            " h",
        ),
        (
            "room_temp",
            climate.actual_c if climate else None,
            conditions.room_temp_min_c,
            conditions.room_temp_max_c,
            " C",
        ),
        (
            "humidity",
            climate.humidity_pct if climate else None,
            conditions.humidity_min_pct,
            conditions.humidity_max_pct,
            " %",
        ),
    )
    for name, observed, minimum, maximum, unit in bounds:
        check = _bound_check(name, observed, minimum, maximum, unit)
        if check is not None:
            checks.append(check)
    if conditions.window_open is not None:
        window = climate.window_open if climate else None
        expected = "open" if conditions.window_open else "closed"
        if window is None:
            outcome, observed_text = CheckOutcome.UNKNOWN, None
        else:
            observed_text = "open" if window else "closed"
            matches = window == conditions.window_open
            outcome = CheckOutcome.MET if matches else CheckOutcome.VIOLATED
        checks.append(
            ConditionCheck(
                condition="window", expected=expected, observed=observed_text, outcome=outcome
            )
        )
    return checks
