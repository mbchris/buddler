"""Integration fixtures: migrated test database, per-test rollback (fastapi-testing rules)."""

from collections.abc import AsyncIterator

import pytest
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from buddler.db import downgrade_database, upgrade_database


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "tests/integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)


class IntegrationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    test_database_url: str
    test_mcp_ro_database_url: str


@pytest.fixture(scope="session")
def integration_settings() -> IntegrationSettings:
    return IntegrationSettings()


@pytest.fixture(scope="session")
def migrated_database(integration_settings: IntegrationSettings) -> str:
    url = integration_settings.test_database_url
    # Reset to a clean schema so migrations are exercised from scratch each session.
    downgrade_database(url, "base")
    upgrade_database(url)
    return url


@pytest.fixture
async def engine(migrated_database: str) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(migrated_database)
    yield engine
    await engine.dispose()


@pytest.fixture
async def session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    async with engine.connect() as connection:
        transaction = await connection.begin()
        async with AsyncSession(
            bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False
        ) as session:
            yield session
        await transaction.rollback()


@pytest.fixture
async def mcp_ro_engine(
    integration_settings: IntegrationSettings, migrated_database: str
) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(integration_settings.test_mcp_ro_database_url)
    yield engine
    await engine.dispose()
