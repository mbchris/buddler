"""In-process background execution of workflow runs (design §1: no job queue).

Runs are pre-created as PENDING so the API can return the id immediately; the
workflow itself records RUNNING/SUCCEEDED/FAILED. On shutdown, tasks are cancelled
and their runs marked FAILED ("cancelled") by `traced_run`.
"""

import asyncio
import logging
import uuid
from collections.abc import Coroutine
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from buddler.agents.state import ReviewMode
from buddler.config import Settings
from buddler.models.enums import RunKind
from buddler.services.resolve_service import ResolveRequest, run_resolve
from buddler.services.run_service import create_pending_run
from buddler.services.train_service import TrainRequest, run_training

logger = logging.getLogger(__name__)


class RunManager:
    def __init__(
        self, settings: Settings, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        self._settings = settings
        self._factory = session_factory
        self._tasks: dict[uuid.UUID, asyncio.Task[None]] = {}

    @property
    def active_run_ids(self) -> list[uuid.UUID]:
        return list(self._tasks)

    async def start_train(self, request: TrainRequest, review_mode: ReviewMode) -> uuid.UUID:
        if review_mode is ReviewMode.INTERACTIVE:
            raise ValueError("Interactive review is CLI-only; use auto or defer.")
        run_id = await create_pending_run(self._factory, RunKind.TRAIN, request.symptom)
        self._spawn(
            run_id,
            run_training(
                request,
                review_mode,
                settings=self._settings,
                factory=self._factory,
                run_id=run_id,
            ),
        )
        return run_id

    async def start_resolve(self, request: ResolveRequest) -> uuid.UUID:
        run_id = await create_pending_run(self._factory, RunKind.RESOLVE, request.situation)
        self._spawn(
            run_id,
            run_resolve(request, settings=self._settings, factory=self._factory, run_id=run_id),
        )
        return run_id

    def _spawn(self, run_id: uuid.UUID, workflow: Coroutine[Any, Any, object]) -> None:
        self._tasks[run_id] = asyncio.create_task(self._execute(run_id, workflow))

    async def _execute(self, run_id: uuid.UUID, workflow: Coroutine[Any, Any, object]) -> None:
        try:
            await workflow
        except Exception:
            # Status and error are already persisted by traced_run; keep the server alive.
            logger.exception("Run %s failed", run_id)
        finally:
            self._tasks.pop(run_id, None)

    async def join(self, run_id: uuid.UUID) -> None:
        """Wait for a run started by this manager (tests, graceful shutdown)."""
        task = self._tasks.get(run_id)
        if task is not None:
            await asyncio.wait([task])

    async def shutdown(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
