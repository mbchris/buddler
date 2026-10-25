# 0005 — Weather from Open-Meteo

## Status
Accepted (2026-10-25)

## Context
Decisions depend on outdoor temperature and whether sunshine is expected in the next hours.

## Decision
- `weather/open_meteo.py` calls `https://api.open-meteo.com/v1/forecast` (no API key) for the
  configured coordinates: current temperature and cloud cover, hourly `sunshine_duration` summed
  over `WEATHER_FORECAST_HOURS`.
- `weather/fake.py` (`WEATHER_PROVIDER=fake`, `FAKE_WEATHER=cloudy|sunny`) for tests and demos.
- Failures degrade the observation to unknown (traced); the run continues.

## Consequences
- Requires outbound internet for real runs; only coordinates are sent.
