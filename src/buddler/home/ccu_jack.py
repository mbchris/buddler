"""CCU-Jack VEAP client (ADR 0004).

Process values live at `/device/{address}/{channel}/{parameter}/~pv` as JSON
`{"v": value, "ts": ms, "s": status}`; a PUT with `{"v": value}` writes one.
Navigation resources (`/room`, `/device/{address}/{channel}`) list children in `~links`.
"""

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass

import httpx2

from buddler.config import RoomConfig
from buddler.home.ports import HomeError, RoomClimate, UnknownRoomError

logger = logging.getLogger(__name__)

ACTUAL_TEMPERATURE = "ACTUAL_TEMPERATURE"
SET_POINT_TEMPERATURE = "SET_POINT_TEMPERATURE"
HUMIDITY = "HUMIDITY"
WINDOW_STATE = "WINDOW_STATE"
CONTACT_STATE = "STATE"
CLIMATE_PARAMETERS = frozenset({ACTUAL_TEMPERATURE, SET_POINT_TEMPERATURE})

JsonValue = float | int | bool | str | None


@dataclass(frozen=True)
class DiscoveredChannel:
    room: str
    address: str
    title: str
    parameters: list[str]

    @property
    def is_climate(self) -> bool:
        return set(self.parameters) >= CLIMATE_PARAMETERS


def _pv_path(channel: str, parameter: str) -> str:
    address, number = channel.split(":")
    return f"/device/{address}/{number}/{parameter}/~pv"


def _resolve(parent: str, href: str) -> str:
    return href if href.startswith("/") else f"{parent.rstrip('/')}/{href}"


def _as_float(value: JsonValue) -> float | None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return None
    return float(value)


def _is_open(value: JsonValue) -> bool | None:
    # WINDOW_STATE 0/1, HmIP contacts 0 closed / 1 tilted|open / 2 open, classic HM booleans.
    if value is None or isinstance(value, str):
        return None
    return bool(value)


class CcuJackHome:
    def __init__(self, client: httpx2.AsyncClient, rooms: Mapping[str, RoomConfig]) -> None:
        self._client = client
        self._rooms = dict(rooms)
        self.name = f"ccu-jack {client.base_url}"

    def rooms(self) -> dict[str, list[str]]:
        return {name: list(room.aliases) for name, room in self._rooms.items()}

    def _room(self, room: str) -> RoomConfig:
        try:
            return self._rooms[room]
        except KeyError as err:
            raise UnknownRoomError(f"Room {room!r} is not in HOMEMATIC_ROOMS.") from err

    async def _get_json(self, path: str) -> Mapping[str, object]:
        try:
            response = await self._client.get(path)
            response.raise_for_status()
            body = response.json()
        except (httpx2.HTTPError, ValueError) as err:
            raise HomeError(f"CCU-Jack GET {path} failed: {err}") from err
        if not isinstance(body, dict):
            raise HomeError(f"CCU-Jack GET {path}: unexpected body")
        return body

    async def read_value(self, channel: str, parameter: str) -> JsonValue:
        """None when the channel has no such parameter (404)."""
        path = _pv_path(channel, parameter)
        try:
            response = await self._client.get(path)
            if response.status_code == httpx2.codes.NOT_FOUND:
                return None
            response.raise_for_status()
            body = response.json()
        except (httpx2.HTTPError, ValueError) as err:
            raise HomeError(f"CCU-Jack read {channel} {parameter} failed: {err}") from err
        value = body.get("v") if isinstance(body, dict) else None
        if isinstance(value, float | int | bool | str) or value is None:
            return value
        return None

    async def read_climate(self, room: str) -> RoomClimate:
        config = self._room(room)
        window_channel, window_parameter = (
            (config.window, CONTACT_STATE) if config.window else (config.climate, WINDOW_STATE)
        )
        actual, setpoint, humidity, window = await asyncio.gather(
            self.read_value(config.climate, ACTUAL_TEMPERATURE),
            self.read_value(config.climate, SET_POINT_TEMPERATURE),
            self.read_value(config.climate, HUMIDITY),
            self.read_value(window_channel, window_parameter),
        )
        return RoomClimate(
            room=room,
            actual_c=_as_float(actual),
            setpoint_c=_as_float(setpoint),
            humidity_pct=_as_float(humidity),
            window_open=_is_open(window),
        )

    async def set_setpoint(self, room: str, value_c: float) -> None:
        path = _pv_path(self._room(room).climate, SET_POINT_TEMPERATURE)
        try:
            response = await self._client.put(path, json={"v": value_c})
            response.raise_for_status()
        except httpx2.HTTPError as err:
            raise HomeError(f"CCU-Jack write {room} setpoint failed: {err}") from err
        logger.info("Set %s setpoint to %.1f °C", room, value_c)

    async def _links(self, path: str, rel: str) -> list[tuple[str, str]]:
        body = await self._get_json(path)
        links = body.get("~links", [])
        found: list[tuple[str, str]] = []
        if isinstance(links, list):
            for link in links:
                if isinstance(link, dict) and link.get("rel") == rel and "href" in link:
                    href = _resolve(path, str(link["href"]))
                    found.append((href, str(link.get("title") or href)))
        return found

    async def discover(self) -> list[DiscoveredChannel]:
        """CCU rooms with their channels and parameter names (helps fill HOMEMATIC_ROOMS)."""
        channels: list[DiscoveredChannel] = []
        for room_path, room_title in await self._links("/room", "room"):
            for channel_path, channel_title in await self._links(room_path, "channel"):
                parameters = [
                    href.rsplit("/", 1)[-1]
                    for href, _ in await self._links(channel_path, "parameter")
                ]
                parts = channel_path.strip("/").split("/")
                address = f"{parts[-2]}:{parts[-1]}" if len(parts) >= 3 else channel_path
                channels.append(
                    DiscoveredChannel(
                        room=room_title,
                        address=address,
                        title=channel_title,
                        parameters=parameters,
                    )
                )
        return channels
