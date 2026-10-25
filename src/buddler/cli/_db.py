"""Engine lifecycle for one CLI command."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from buddler.config import get_settings
from buddler.db import create_engine, create_session_factory


@asynccontextmanager
async def cli_session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_engine(get_settings())
    try:
        yield create_session_factory(engine)
    finally:
        await engine.dispose()
