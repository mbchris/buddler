"""Canned weather for tests and offline demos (`WEATHER_PROVIDER=fake`)."""

from dataclasses import dataclass
from typing import Literal

from buddler.weather.ports import WeatherObservation

FakeScenario = Literal["cloudy", "sunny"]

_SCENARIOS: dict[str, tuple[float, float, float]] = {
    # outdoor °C, cloud cover %, sunshine hours in the forecast window
    "cloudy": (4.0, 95.0, 0.0),
    "sunny": (9.0, 10.0, 4.5),
}


@dataclass
class FakeWeather:
    scenario: FakeScenario = "cloudy"
    forecast_hours: int = 6

    async def current(self) -> WeatherObservation:
        temp, clouds, sun = _SCENARIOS[self.scenario]
        return WeatherObservation(
            outdoor_temp_c=temp,
            cloud_cover_pct=clouds,
            sun_hours_forecast=min(sun, float(self.forecast_hours)),
            forecast_hours=self.forecast_hours,
            source=f"fake ({self.scenario})",
        )
