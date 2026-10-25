"""Read-only home overview for the CLI and API (no workflow, no tracing)."""

import asyncio

from pydantic import BaseModel

from buddler.home.ports import HomeError, HomePort, RoomClimate


class RoomStatus(BaseModel):
    room: str
    aliases: list[str]
    writable: bool
    climate: RoomClimate | None
    error: str | None = None


async def room_overview(home: HomePort, writable_rooms: list[str]) -> list[RoomStatus]:
    rooms = home.rooms()

    async def read(room: str) -> RoomStatus:
        status = RoomStatus(
            room=room, aliases=rooms[room], writable=room in writable_rooms, climate=None
        )
        try:
            return status.model_copy(update={"climate": await home.read_climate(room)})
        except HomeError as err:
            return status.model_copy(update={"error": str(err)})

    return list(await asyncio.gather(*(read(room) for room in sorted(rooms))))
