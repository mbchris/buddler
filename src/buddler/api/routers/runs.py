"""Start workflow runs and observe them (status, events, live SSE stream)."""

import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from buddler.api.dependencies import FactoryDep, RunManagerDep, SessionDep, SettingsDep
from buddler.models.enums import RunStatus
from buddler.schemas.api import ResolveRunCreate, RunAccepted, RunCreate, RunDetail
from buddler.schemas.trace import RunOut, TraceEventOut
from buddler.services.resolve_service import ResolveRequest
from buddler.services.run_query_service import (
    Heartbeat,
    RunNotFoundError,
    RunQueryService,
    StreamEnd,
    follow_run,
)
from buddler.services.train_service import TrainRequest

router = APIRouter(prefix="/runs", tags=["runs"])

SSE_MEDIA_TYPE = "text/event-stream"


def _not_found(err: LookupError) -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, str(err))


@router.post("", response_model=RunAccepted, status_code=status.HTTP_202_ACCEPTED)
async def start_run(body: RunCreate, request: Request, manager: RunManagerDep) -> RunAccepted:
    """Start a resolve or train run in the background; poll or stream it by id."""
    if isinstance(body, ResolveRunCreate):
        run_id = await manager.start_resolve(
            ResolveRequest(
                situation=body.situation,
                room=body.room,
                at=body.at,
                dry_run=body.dry_run,
                use_mcp=body.use_mcp,
            )
        )
    else:
        run_id = await manager.start_train(
            TrainRequest(symptom=body.symptom, hint=body.hint), body.review_mode
        )
    return RunAccepted(
        run_id=run_id,
        status=RunStatus.PENDING,
        status_url=str(request.url_for("get_run", run_id=run_id)),
        stream_url=str(request.url_for("stream_run", run_id=run_id)),
    )


@router.get("", response_model=list[RunOut], status_code=status.HTTP_200_OK)
async def list_runs(
    session: SessionDep, limit: Annotated[int, Query(ge=1, le=200)] = 20
) -> list[RunOut]:
    return await RunQueryService(session).list_runs(limit)


@router.get("/{run_id}", response_model=RunDetail, status_code=status.HTTP_200_OK)
async def get_run(run_id: uuid.UUID, session: SessionDep) -> RunDetail:
    service = RunQueryService(session)
    try:
        run = await service.get_run(run_id)
    except RunNotFoundError as err:
        raise _not_found(err) from err
    return RunDetail(run=run, resolution=await service.get_resolution(run_id))


@router.get("/{run_id}/events", response_model=list[TraceEventOut], status_code=status.HTTP_200_OK)
async def list_events(
    run_id: uuid.UUID,
    session: SessionDep,
    after_seq: Annotated[int, Query(ge=0)] = 0,
) -> list[TraceEventOut]:
    try:
        return await RunQueryService(session).events(run_id, after_seq)
    except RunNotFoundError as err:
        raise _not_found(err) from err


@router.get(
    "/{run_id}/stream",
    response_class=StreamingResponse,
    status_code=status.HTTP_200_OK,
    responses={200: {"content": {SSE_MEDIA_TYPE: {}}, "description": "Server-sent trace events"}},
)
async def stream_run(
    run_id: uuid.UUID,
    session: SessionDep,
    factory: FactoryDep,
    settings: SettingsDep,
    last_event_id: Annotated[int | None, Header()] = None,
    after_seq: Annotated[int, Query(ge=0)] = 0,
) -> StreamingResponse:
    """SSE: `trace` events (id = seq, resumable via Last-Event-ID), then one `end` event."""
    try:
        await RunQueryService(session).get_run(run_id)
    except RunNotFoundError as err:
        raise _not_found(err) from err
    start = last_event_id if last_event_id is not None else after_seq

    async def events() -> AsyncIterator[str]:
        async for item in follow_run(
            factory,
            run_id,
            after_seq=start,
            poll_interval_s=settings.sse_poll_interval_s,
            heartbeat_s=settings.sse_heartbeat_s,
        ):
            if isinstance(item, Heartbeat):
                yield ": keep-alive\n\n"
            elif isinstance(item, StreamEnd):
                yield f"event: end\ndata: {item.run.model_dump_json()}\n\n"
            else:
                yield f"id: {item.seq}\nevent: trace\ndata: {item.model_dump_json()}\n\n"

    return StreamingResponse(
        events(),
        media_type=SSE_MEDIA_TYPE,
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
