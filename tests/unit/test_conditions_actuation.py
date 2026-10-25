from datetime import UTC, datetime, time, timedelta

import pytest
from pydantic import ValidationError

from buddler.home.ports import RoomClimate
from buddler.knowledge.actuation import (
    ActuationBounds,
    GuardDecision,
    check_runtime,
    check_static,
    round_setpoint,
)
from buddler.knowledge.conditions import (
    ActionConditions,
    ActionParams,
    CheckOutcome,
    ConditionCheck,
    Observations,
    evaluate_conditions,
    in_window,
    validate_params,
)
from buddler.models.enums import ActionKind
from buddler.weather.ports import WeatherObservation

BOUNDS = ActuationBounds(
    rooms=frozenset({"living room"}),
    setpoint_min_c=16,
    setpoint_max_c=24,
    max_delta_c=2,
    max_writes_per_run=1,
    cooldown=timedelta(minutes=30),
)
MET = [ConditionCheck(condition="x", expected="", observed="", outcome=CheckOutcome.MET)]


def observations(hour: int = 13, *, climate: bool = True, weather: bool = True) -> Observations:
    return Observations(
        local_time=datetime(2026, 11, 10, hour, 30, tzinfo=UTC),
        weather=WeatherObservation(
            outdoor_temp_c=4,
            cloud_cover_pct=90,
            sun_hours_forecast=0.5,
            forecast_hours=6,
            source="test",
        )
        if weather
        else None,
        climate=RoomClimate(
            room="living room", actual_c=19.5, setpoint_c=20, humidity_pct=50, window_open=False
        )
        if climate
        else None,
    )


@pytest.mark.parametrize(
    ("now", "start", "end", "inside"),
    [
        (time(13), time(6), time(22), True),
        (time(22), time(6), time(22), False),
        (time(2), time(22), time(6), True),
        (time(23, 59), time(22), time(6), True),
        (time(6), time(22), time(6), False),
    ],
)
def test_time_window_wraps_midnight(now: time, start: time, end: time, inside: bool) -> None:
    assert in_window(now, start, end) is inside


def test_conditions_report_met_violated_and_unknown() -> None:
    conditions = ActionConditions(
        time_from=time(6),
        time_to=time(22),
        sun_hours_max=1,
        room_temp_max_c=19,
        window_open=False,
    )

    checks = {c.condition: c.outcome for c in evaluate_conditions(conditions, observations())}
    missing = evaluate_conditions(conditions, observations(climate=False))

    assert checks == {
        "time_window": CheckOutcome.MET,
        "sun_hours": CheckOutcome.MET,
        "room_temp": CheckOutcome.VIOLATED,
        "window": CheckOutcome.MET,
    }
    assert {c.condition for c in missing if c.outcome is CheckOutcome.UNKNOWN} == {
        "room_temp",
        "window",
    }


def test_conditions_reject_unknown_fields_and_half_windows() -> None:
    with pytest.raises(ValidationError):
        ActionConditions.model_validate({"moon_phase": "full"})
    with pytest.raises(ValidationError):
        ActionConditions(time_from=time(6))


@pytest.mark.parametrize(
    ("kind", "params"),
    [
        (ActionKind.HEATING_DELTA, ActionParams()),
        (ActionKind.HEATING_SETPOINT, ActionParams(delta_c=1)),
        (ActionKind.ADVICE, ActionParams(setpoint_c=21)),
    ],
)
def test_params_must_fit_kind(kind: ActionKind, params: ActionParams) -> None:
    with pytest.raises(ValueError):
        validate_params(kind, params)


def static(
    kind: ActionKind = ActionKind.HEATING_DELTA,
    params: ActionParams | None = None,
    *,
    room: str | None = "living room",
    current: float | None = 20,
    checks: list[ConditionCheck] = MET,
) -> GuardDecision:
    params = params or ActionParams(delta_c=1)
    return check_static(kind, params, room=room, current_c=current, checks=checks, bounds=BOUNDS)


def test_guard_allows_bounded_change() -> None:
    decision = static()

    assert decision.allowed and decision.target_c == 21


@pytest.mark.parametrize(
    ("decision", "reason"),
    [
        (static(room="office"), "ACTUATION_ROOMS"),
        (static(room=None), "no room"),
        (static(current=None), "setpoint unknown"),
        (static(params=ActionParams(delta_c=3)), "max delta"),
        (static(current=23.5), "outside"),
        (static(ActionKind.HEATING_SETPOINT, ActionParams(setpoint_c=20)), "already"),
        (
            static(
                checks=[
                    ConditionCheck(
                        condition="window",
                        expected="closed",
                        observed=None,
                        outcome=CheckOutcome.UNKNOWN,
                    )
                ]
            ),
            "unknown",
        ),
    ],
)
def test_guard_rejects(decision: GuardDecision, reason: str) -> None:
    assert not decision.allowed
    assert reason in decision.reason


def test_runtime_limits_writes_and_cooldown() -> None:
    now = datetime(2026, 11, 10, 13, tzinfo=UTC)
    allowed = static()

    def runtime(writes: int, last: datetime | None) -> GuardDecision:
        return check_runtime(
            allowed, writes_this_run=writes, last_write_at=last, now=now, bounds=BOUNDS
        )

    assert runtime(0, None).allowed
    assert runtime(0, now - timedelta(hours=1)).allowed
    assert "per run" in runtime(1, None).reason
    assert "cooldown 30 min" in runtime(0, now - timedelta(minutes=5)).reason


def test_setpoints_round_to_half_degrees() -> None:
    assert round_setpoint(20.74) == 20.5
    assert round_setpoint(20.76) == 21.0
