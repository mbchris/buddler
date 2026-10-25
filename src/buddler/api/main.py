"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, status
from pydantic import BaseModel

from buddler import __version__
from buddler.api.dependencies import require_api_key
from buddler.api.routers import graphs, home, knowledge, runs, ui
from buddler.api.run_manager import RunManager
from buddler.config import Settings, get_settings
from buddler.db import create_engine, create_session_factory, ensure_embedding_dim


class HealthResponse(BaseModel):
    status: str
    version: str


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    ensure_embedding_dim(settings)
    # Created eagerly (connections are lazy) so tests work without running the lifespan.
    engine = create_engine(settings)
    factory = create_session_factory(engine)
    run_manager = RunManager(settings, factory)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        await run_manager.shutdown()
        await engine.dispose()

    app = FastAPI(title="Buddler", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.session_factory = factory
    app.state.run_manager = run_manager

    @app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    app.include_router(ui.router)
    app.mount("/ui/static", ui.RevalidatedStaticFiles(directory=ui.STATIC_DIR), name="ui-static")

    protected = [Depends(require_api_key)]
    for router in (runs.router, knowledge.router, home.router, graphs.router):
        app.include_router(router, dependencies=protected)
    return app


def app() -> FastAPI:
    """uvicorn factory entry point: `uvicorn --factory buddler.api.main:app`."""
    return create_app()
