"""Alembic environment (async). URL: `sqlalchemy.url` override, else Settings.database_url."""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from buddler.config import get_settings
from buddler.models import APP_SCHEMA, Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _database_url() -> str:
    return config.get_main_option("sqlalchemy.url") or str(get_settings().database_url)


def _include_name(name: str | None, type_: str, parent_names: object) -> bool:
    # Only diff our own schema; `knowledge` views are managed by hand-written migrations.
    if type_ == "schema":
        return name == APP_SCHEMA
    return True


def _configure(connection: Connection | None = None) -> None:
    context.configure(
        connection=connection,
        url=None if connection else _database_url(),
        target_metadata=target_metadata,
        include_schemas=True,
        include_name=_include_name,
        compare_type=True,
        literal_binds=connection is None,
    )


def run_migrations_offline() -> None:
    _configure()
    with context.begin_transaction():
        context.run_migrations()


def _run_sync(connection: Connection) -> None:
    _configure(connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(_database_url())
    async with engine.connect() as connection:
        await connection.run_sync(_run_sync)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
