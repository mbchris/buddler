"""CCU-Jack and Open-Meteo adapters against in-memory httpx2 transports (no network)."""

import json
from collections.abc import Callable

import httpx2
import pytest

from buddler.config import RoomConfig
from buddler.home.ccu_jack import CcuJackHome
from buddler.home.factory import ccu_jack_client
from buddler.home.ports import HomeError, UnknownRoomError
from buddler.weather.open_meteo import OpenMeteoWeather
from buddler.weather.ports import WeatherError
from tests.unit.workflow_fakes import make_settings

Handler = Callable[[httpx2.Request], httpx2.Response]
PV = {
    "/device/THERMO1/1/ACTUAL_TEMPERATURE/~pv": {"v": 19.4, "ts": 1, "s": 0},
    "/device/THERMO1/1/SET_POINT_TEMPERATURE/~pv": {"v": 20.0, "ts": 1, "s": 0},
    "/device/THERMO1/1/WINDOW_STATE/~pv": {"v": 0, "ts": 1, "s": 0},
    "/device/CONTACT1/1/STATE/~pv": {"v": 2, "ts": 1, "s": 0},
}


def ccu(handler: Handler, rooms: dict[str, RoomConfig]) -> tuple[CcuJackHome, httpx2.AsyncClient]:
    settings = make_settings(
        homematic_endpoint="https://ccu.test:2122",
        homematic_username="user",
        homematic_password="secret",
    )
    client = ccu_jack_client(settings, httpx2.MockTransport(handler))
    return CcuJackHome(client, rooms), client


def pv_handler(writes: list[tuple[str, object]]) -> Handler:
    def handle(request: httpx2.Request) -> httpx2.Response:
        assert request.headers["authorization"].startswith("Basic ")
        if request.method == "PUT":
            writes.append((request.url.path, json.loads(request.content)))
            return httpx2.Response(200)
        body = PV.get(request.url.path)
        return httpx2.Response(200, json=body) if body else httpx2.Response(404)

    return handle


async def test_read_climate_maps_parameters_and_missing_ones() -> None:
    home, client = ccu(pv_handler([]), {"living room": RoomConfig(climate="THERMO1:1")})
    async with client:
        climate = await home.read_climate("living room")

    assert (climate.actual_c, climate.setpoint_c) == (19.4, 20.0)
    assert climate.humidity_pct is None  # 404: parameter not on this channel
    assert climate.window_open is False


async def test_separate_window_contact_is_used() -> None:
    rooms = {"office": RoomConfig(climate="THERMO1:1", window="CONTACT1:1")}
    home, client = ccu(pv_handler([]), rooms)
    async with client:
        climate = await home.read_climate("office")

    assert climate.window_open is True


async def test_set_setpoint_puts_process_value() -> None:
    writes: list[tuple[str, object]] = []
    home, client = ccu(pv_handler(writes), {"living room": RoomConfig(climate="THERMO1:1")})
    async with client:
        await home.set_setpoint("living room", 21.0)

    assert writes == [("/device/THERMO1/1/SET_POINT_TEMPERATURE/~pv", {"v": 21.0})]


async def test_errors_become_home_errors() -> None:
    home, client = ccu(lambda request: httpx2.Response(500), {"x": RoomConfig(climate="A:1")})
    async with client:
        with pytest.raises(HomeError):
            await home.read_climate("x")
        with pytest.raises(UnknownRoomError):
            await home.read_climate("garage")


async def test_discover_follows_room_and_channel_links() -> None:
    pages = {
        "/room": {"~links": [{"rel": "room", "href": "1234", "title": "Wohnzimmer"}]},
        "/room/1234": {
            "~links": [{"rel": "channel", "href": "/device/THERMO1/1", "title": "Thermostat"}]
        },
        "/device/THERMO1/1": {
            "~links": [
                {"rel": "parameter", "href": "ACTUAL_TEMPERATURE"},
                {"rel": "parameter", "href": "SET_POINT_TEMPERATURE"},
                {"rel": "device", "href": ".."},
            ]
        },
    }
    home, client = ccu(lambda r: httpx2.Response(200, json=pages[r.url.path]), {})
    async with client:
        [channel] = await home.discover()

    assert (channel.room, channel.address, channel.is_climate) == ("Wohnzimmer", "THERMO1:1", True)


def open_meteo(handler: Handler) -> tuple[OpenMeteoWeather, httpx2.AsyncClient]:
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    weather = OpenMeteoWeather(
        client,
        url="https://meteo.test/v1/forecast",
        latitude=52.5,
        longitude=13.4,
        forecast_hours=3,
    )
    return weather, client


async def test_open_meteo_sums_sunshine_over_forecast_window() -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        assert request.url.params["forecast_hours"] == "3"
        return httpx2.Response(
            200,
            json={
                "current": {"temperature_2m": 6.2, "cloud_cover": 40},
                "hourly": {"sunshine_duration": [3600, 1800, None, 3600]},
            },
        )

    weather, client = open_meteo(handle)
    async with client:
        observation = await weather.current()

    assert observation.outdoor_temp_c == 6.2
    assert observation.sun_hours_forecast == 1.5


async def test_open_meteo_failure_is_weather_error() -> None:
    weather, client = open_meteo(lambda request: httpx2.Response(503))
    async with client:
        with pytest.raises(WeatherError):
            await weather.current()
