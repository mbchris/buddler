"""Shared FastAPI dependencies: settings, sessions, run manager, API-key auth."""

import secrets
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from buddler.api.run_manager import RunManager
from buddler.config import Settings
from buddler.db import session_scope

_bearer = HTTPBearer(auto_error=False, description="Static API key (API_KEY).")


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    factory: async_sessionmaker[AsyncSession] = request.app.state.session_factory
    return factory


def get_run_manager(request: Request) -> RunManager:
    manager: RunManager = request.app.state.run_manager
    return manager


async def get_session(
    factory: Annotated[async_sessionmaker[AsyncSession], Depends(get_session_factory)],
) -> AsyncIterator[AsyncSession]:
    """One session per request; committed when the handler succeeds."""
    async with session_scope(factory) as session:
        yield session


def require_api_key(
    settings: Annotated[Settings, Depends(get_app_settings)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    """Static bearer key (design §9: local demonstrator, JWT intentionally deferred)."""
    expected = settings.api_key.get_secret_value()
    provided = credentials.credentials if credentials else ""
    if not expected or not secrets.compare_digest(provided.encode(), expected.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
FactoryDep = Annotated[async_sessionmaker[AsyncSession], Depends(get_session_factory)]
RunManagerDep = Annotated[RunManager, Depends(get_run_manager)]
