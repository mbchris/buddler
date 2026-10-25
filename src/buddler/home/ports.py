"""Homematic port: the only way workflows read or change the home."""

from typing import Protocol

from pydantic import BaseModel


class HomeError(RuntimeError):
    """CCU unreachable, unknown room/channel, or a rejected write."""


class UnknownRoomError(HomeError):
    pass


class RoomClimate(BaseModel):
    room: str
    # None where the channel does not provide the parameter.
    actual_c: float | None
    setpoint_c: float | None
    humidity_pct: float | None = None
    window_open: bool | None = None


class HomePort(Protocol):
    # Shown in traces ("ccu-jack https://…", "fake").
    name: str

    def rooms(self) -> dict[str, list[str]]:
        """Configured room names with their aliases."""
        ...

    async def read_climate(self, room: str) -> RoomClimate: ...

    async def set_setpoint(self, room: str, value_c: float) -> None: ...
