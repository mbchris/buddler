"""Open-Meteo forecast client (ADR 0005). No API key; only coordinates are sent."""

from collections.abc import Sequence

import httpx2

from buddler.weather.ports import WeatherError, WeatherObservation

SECONDS_PER_HOUR = 3600


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _sun_hours(values: object, hours: int) -> float | None:
    if not isinstance(values, Sequence) or isinstance(values, str):
        return None
    seconds = [_number(v) for v in values[:hours]]
    known = [s for s in seconds if s is not None]
    return round(sum(known) / SECONDS_PER_HOUR, 2) if known else None


class OpenMeteoWeather:
    def __init__(
        self,
        client: httpx2.AsyncClient,
        *,
        url: str,
        latitude: float,
        longitude: float,
        forecast_hours: int,
    ) -> None:
        self._client = client
        self._url = url
        self._params: dict[str, str | float | int] = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,cloud_cover",
            "hourly": "sunshine_duration",
            "forecast_hours": forecast_hours,
            "timezone": "auto",
        }
        self._hours = forecast_hours

    async def current(self) -> WeatherObservation:
        try:
            response = await self._client.get(self._url, params=self._params)
            response.raise_for_status()
            body = response.json()
        except (httpx2.HTTPError, ValueError) as err:
            raise WeatherError(f"Open-Meteo request failed: {err}") from err
        if not isinstance(body, dict):
            raise WeatherError("Open-Meteo: unexpected response body")
        current = body.get("current") or {}
        hourly = body.get("hourly") or {}
        if not isinstance(current, dict) or not isinstance(hourly, dict):
            raise WeatherError("Open-Meteo: unexpected response body")
        return WeatherObservation(
            outdoor_temp_c=_number(current.get("temperature_2m")),
            cloud_cover_pct=_number(current.get("cloud_cover")),
            sun_hours_forecast=_sun_hours(hourly.get("sunshine_duration"), self._hours),
            forecast_hours=self._hours,
            source="open-meteo",
        )
