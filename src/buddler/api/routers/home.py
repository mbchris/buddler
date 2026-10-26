"""Read-only view of the configured rooms and their current climate."""

from fastapi import APIRouter, status

from buddler.api.dependencies import SettingsDep
from buddler.home.factory import open_home
from buddler.services.home_service import RoomStatus, room_overview

router = APIRouter(prefix="/home", tags=["home"])


@router.get("/rooms", response_model=list[RoomStatus], status_code=status.HTTP_200_OK)
async def list_rooms(settings: SettingsDep) -> list[RoomStatus]:
    async with open_home(settings) as home:
        return await room_overview(home, settings.actuation_rooms)
