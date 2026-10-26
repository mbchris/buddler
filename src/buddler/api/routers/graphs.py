"""Workflow structure for the demonstrator view."""

from dataclasses import asdict

from fastapi import APIRouter, status

from buddler.agents.graphs.explain import explanation
from buddler.agents.graphs.resolve import RESOLVE_NODE_ORDER, resolve_graph_mermaid
from buddler.agents.graphs.train import TRAIN_NODE_ORDER, train_graph_mermaid
from buddler.api.dependencies import SettingsDep
from buddler.models.enums import RunKind
from buddler.schemas.api import GlossaryTermOut, GraphOut, GraphStepOut

router = APIRouter(prefix="/graphs", tags=["graphs"])

_MERMAID = {RunKind.TRAIN: train_graph_mermaid, RunKind.RESOLVE: resolve_graph_mermaid}
_ORDER = {RunKind.TRAIN: TRAIN_NODE_ORDER, RunKind.RESOLVE: RESOLVE_NODE_ORDER}


@router.get("/{kind}", response_model=GraphOut, status_code=status.HTTP_200_OK)
async def get_graph(kind: RunKind, settings: SettingsDep) -> GraphOut:
    explained = explanation(kind, settings)
    return GraphOut(
        kind=kind,
        mermaid=_MERMAID[kind](),
        overview=explained.overview,
        glossary=[
            GlossaryTermOut(term=term, definition=definition)
            for term, definition in explained.glossary
        ],
        steps=[GraphStepOut(node=node, **asdict(explained.steps[node])) for node in _ORDER[kind]],
    )
