"""Knowledge curation: list symptoms with actions, review entries (AGENTS.md §4)."""

import uuid
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from buddler.api.dependencies import SessionDep
from buddler.models.enums import KnowledgeStatus
from buddler.schemas.knowledge import KnowledgeReview, SymptomOut
from buddler.services.knowledge_service import KnowledgeNotFoundError, KnowledgeService

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.get("/symptoms", response_model=list[SymptomOut], status_code=status.HTTP_200_OK)
async def list_symptoms(
    session: SessionDep,
    status_filter: Annotated[KnowledgeStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[SymptomOut]:
    """Newest first; `status` matches symptoms or any of their actions."""
    return await KnowledgeService(session).list_symptoms(status_filter, limit)


@router.patch("/symptoms/{symptom_id}", response_model=SymptomOut, status_code=status.HTTP_200_OK)
async def review_symptom(
    symptom_id: uuid.UUID, body: KnowledgeReview, session: SessionDep
) -> SymptomOut:
    try:
        return await KnowledgeService(session).review_symptom(symptom_id, body.status)
    except KnowledgeNotFoundError as err:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(err)) from err


@router.patch("/actions/{action_id}", response_model=SymptomOut, status_code=status.HTTP_200_OK)
async def review_action(
    action_id: uuid.UUID, body: KnowledgeReview, session: SessionDep
) -> SymptomOut:
    """Returns the action's symptom with all its actions."""
    try:
        return await KnowledgeService(session).review_action(action_id, body.status)
    except KnowledgeNotFoundError as err:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(err)) from err
