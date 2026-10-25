"""Build the weather adapter from settings."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx2

from buddler.config import Settings
from buddler.weather.fake import FakeWeather
from buddler.weather.open_meteo import OpenMeteoWeather
from buddler.weather.ports import WeatherPort


@asynccontextmanager
async def open_weather(
    settings: Settings, transport: httpx2.AsyncBaseTransport | None = None
) -> AsyncIterator[WeatherPort]:
    if settings.weather_provider == "fake":
        yield FakeWeather(settings.fake_weather, settings.weather_forecast_hours)
        return
    async with httpx2.AsyncClient(
        timeout=settings.weather_timeout_s, transport=transport, trust_env=False
    ) as client:
        yield OpenMeteoWeather(
            client,
            url=settings.weather_base_url,
            latitude=settings.location_latitude,
            longitude=settings.location_longitude,
            forecast_hours=settings.weather_forecast_hours,
        )
