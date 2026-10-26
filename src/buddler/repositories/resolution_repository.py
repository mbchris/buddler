"""Persistence for resolve results and the device command log."""

import uuid
from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from buddler.models import DeviceCommand, Resolution
from buddler.models.enums import CommandStatus
from buddler.models.run import JsonObject


class ResolutionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(
        self, run_id: uuid.UUID, body: JsonObject, citations: list[JsonObject]
    ) -> uuid.UUID:
        resolution = Resolution(run_id=run_id, body=body, citations=citations)
        self._session.add(resolution)
        await self._session.flush()
        return resolution.id

    async def get_for_run(self, run_id: uuid.UUID) -> Resolution | None:
        return await self._session.scalar(select(Resolution).where(Resolution.run_id == run_id))


class DeviceCommandRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, command: DeviceCommand) -> uuid.UUID:
        self._session.add(command)
        await self._session.flush()
        return command.id

    async def last_applied_at(self, room: str) -> datetime | None:
        query = select(func.max(DeviceCommand.created_at)).where(
            DeviceCommand.room == room, DeviceCommand.status == CommandStatus.APPLIED
        )
        return await self._session.scalar(query)

    async def list_for_run(self, run_id: uuid.UUID) -> Sequence[DeviceCommand]:
        query = (
            select(DeviceCommand)
            .where(DeviceCommand.run_id == run_id)
            .order_by(DeviceCommand.created_at)
        )
        return (await self._session.scalars(query)).all()
