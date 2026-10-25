"""Async engine and session factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from buddler.config import Settings
from buddler.models.base import EMBEDDING_DIM

MIGRATIONS_LOCATION = "buddler:migrations"


def create_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(str(settings.database_url), pool_pre_ping=True)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


@asynccontextmanager
async def session_scope(factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    """One session per unit of work; commits on success, rolls back on error."""
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


class EmbeddingDimensionMismatchError(RuntimeError):
    pass


def ensure_embedding_dim(settings: Settings) -> None:
    """Changing the embedding model's dimension needs a migration + re-embedding (AGENTS.md §2)."""
    if settings.embed_dim != EMBEDDING_DIM:
        msg = (
            f"EMBED_DIM={settings.embed_dim} does not match schema dimension {EMBEDDING_DIM}; "
            "add an Alembic migration and re-embed before switching models."
        )
        raise EmbeddingDimensionMismatchError(msg)


def _alembic_config(database_url: str) -> AlembicConfig:
    config = AlembicConfig()
    config.set_main_option("script_location", MIGRATIONS_LOCATION)
    # ConfigParser interpolation: escape '%' in URL-encoded passwords.
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config


def upgrade_database(database_url: str, revision: str = "head") -> None:
    """Blocking; call via `asyncio.to_thread` from async code."""
    command.upgrade(_alembic_config(database_url), revision)


def downgrade_database(database_url: str, revision: str) -> None:
    command.downgrade(_alembic_config(database_url), revision)
