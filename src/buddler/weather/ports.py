"""Weather port: current conditions and short-term sunshine forecast."""

from typing import Protocol

from pydantic import BaseModel


class WeatherError(RuntimeError):
    pass


class WeatherObservation(BaseModel):
    outdoor_temp_c: float | None
    cloud_cover_pct: float | None
    # Sum of forecast sunshine over the next `forecast_hours`.
    sun_hours_forecast: float | None
    forecast_hours: int
    source: str


class WeatherPort(Protocol):
    async def current(self) -> WeatherObservation: ...
