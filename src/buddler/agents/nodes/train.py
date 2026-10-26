"""Train workflow nodes (design §3); each runs inside a tracer span via `traced_node`."""

import json

from langgraph.types import interrupt
from pydantic import TypeAdapter

from buddler.agents.ports import TrainDeps
from buddler.agents.schemas import ActionProposalDraft, ProposalResponse
from buddler.agents.state import ProposedAction, ReviewDecision, ReviewMode, TrainState
from buddler.agents.tracing import NodeUpdate
from buddler.knowledge.actuation import ActuationBounds, round_setpoint
from buddler.knowledge.conditions import validate_params
from buddler.llm.ports import ChatMessage
from buddler.models.enums import ActionKind, KnowledgeStatus, RunKind, TraceEventType
from buddler.prompts.loader import load_prompt
from buddler.tracing.tracer import SpanResult

PROPOSE_PROMPT = "propose"
PREVIEW_ITEMS = 5
MAX_SYMPTOM_CHARS = 120

_decisions_adapter = TypeAdapter(list[ReviewDecision])


def out_of_bounds(draft: ActionProposalDraft, bounds: ActuationBounds) -> str | None:
    """Reason a proposed device action can never pass the actuation guard, else None."""
    try:
        validate_params(draft.kind, draft.params)
    except ValueError as err:
        return str(err)
    delta = draft.params.delta_c
    if (
        draft.kind is ActionKind.HEATING_DELTA
        and delta is not None
        and abs(delta) > bounds.max_delta_c
    ):
        return f"delta {delta:+g} C exceeds max {bounds.max_delta_c:g} C"
    if draft.kind is ActionKind.HEATING_SETPOINT and draft.params.setpoint_c is not None:
        target = round_setpoint(draft.params.setpoint_c)
        if not bounds.setpoint_min_c <= target <= bounds.setpoint_max_c:
            return f"setpoint {target:g} C outside bounds"
    return None


class TrainNodes:
    def __init__(self, deps: TrainDeps) -> None:
        self._deps = deps
        self._bounds = ActuationBounds.from_settings(deps.settings)

    async def embed(self, state: TrainState, span: SpanResult) -> NodeUpdate:
        result = await self._deps.embedder.embed([state["symptom"]])
        if result.usage is not None:
            await self._deps.tracer.llm_call(
                result.usage,
                f"Embedded '{state['symptom']}'",
                type_=TraceEventType.EMBEDDING_CALL,
                payload={"cache_hits": result.cache_hits},
            )
        span.summary = f"embedded with {self._deps.embedder.model} ({self._deps.embedder.dim} dim)"
        return {"embedding": result.vectors[0]}

    async def match(self, state: TrainState, span: SpanResult) -> NodeUpdate:
        max_distance = self._deps.settings.train_duplicate_max_distance
        existing = await self._deps.knowledge.nearest_any(state["embedding"], max_distance)
        if existing is None:
            span.summary = f"new symptom (nothing within distance {max_distance})"
        else:
            span.summary = (
                f"matches '{existing.text}' ({existing.status}, distance "
                f"{existing.distance:.3f}) with {len(existing.actions)} actions"
            )
        span.payload = {
            "max_distance": max_distance,
            "existing": existing.model_dump(mode="json") if existing else None,
        }
        return {"existing": existing}

    async def propose(self, state: TrainState, span: SpanResult) -> NodeUpdate:
        prompt = load_prompt(RunKind.TRAIN, PROPOSE_PROMPT)
        existing = state.get("existing")
        existing_block = (
            json.dumps(
                {
                    "symptom": existing.text,
                    "status": existing.status,
                    "actions": [
                        a.model_dump(mode="json", exclude={"id"}, exclude_none=True)
                        for a in existing.actions
                    ],
                }
            )
            if existing
            else "none"
        )
        bounds = {
            "setpoint_min_c": self._bounds.setpoint_min_c,
            "setpoint_max_c": self._bounds.setpoint_max_c,
            "max_delta_c": self._bounds.max_delta_c,
        }
        user = (
            f"<symptom>\n{state['symptom']}\n</symptom>\n"
            f"<hint>\n{state.get('hint') or 'none'}\n</hint>\n"
            f"<existing>\n{existing_block}\n</existing>\n"
            f"<bounds>\n{json.dumps(bounds)}\n</bounds>"
        )
        result = await self._deps.chat.complete_structured(
            [
                ChatMessage(role="system", content=prompt.text),
                ChatMessage(role="user", content=user),
            ],
            ProposalResponse,
        )
        response = result.value
        await self._deps.tracer.llm_call(
            result.usage,
            f"Proposed {len(response.actions)} actions for '{response.symptom}'",
            payload={"attempts": result.attempts},
        )
        proposals: list[ProposedAction] = []
        for draft in response.actions:
            reason = out_of_bounds(draft, self._bounds)
            if reason is not None:
                await self._deps.tracer.emit(
                    TraceEventType.ERROR,
                    f"Dropped proposal '{draft.title}': {reason}",
                    payload=draft.model_dump(mode="json"),
                )
                continue
            proposals.append(ProposedAction(ref=len(proposals) + 1, **draft.model_dump()))
        normalized = (response.symptom.strip() or state["symptom"])[:MAX_SYMPTOM_CHARS]
        span.summary = f"{len(proposals)} proposals for '{normalized}'"
        span.payload = {
            "prompt_version": prompt.version,
            "dropped": len(response.actions) - len(proposals),
            "proposals_preview": [p.title for p in proposals[:PREVIEW_ITEMS]],
        }
        return {
            "proposals": proposals,
            "normalized_symptom": normalized,
            "prompt_version": prompt.version,
        }

    async def review(self, state: TrainState, span: SpanResult) -> NodeUpdate:
        proposals = state["proposals"]
        mode = state["review_mode"]
        if not proposals:
            span.summary = "nothing to review"
            return {"decisions": []}
        if mode is ReviewMode.AUTO_CONFIRM:
            decisions = [
                ReviewDecision(ref=p.ref, status=KnowledgeStatus.CONFIRMED) for p in proposals
            ]
        elif mode is ReviewMode.DEFER:
            decisions = [
                ReviewDecision(ref=p.ref, status=KnowledgeStatus.PROPOSED) for p in proposals
            ]
        else:
            answer = interrupt({"proposals": [p.model_dump(mode="json") for p in proposals]})
            decisions = _decisions_adapter.validate_python(answer)
        counts = {s: sum(d.status is s for d in decisions) for s in KnowledgeStatus}
        span.summary = f"{mode} review: " + ", ".join(f"{n} {s}" for s, n in counts.items() if n)
        span.payload = {"mode": mode, "counts": counts}
        return {"decisions": decisions}

    async def persist(self, state: TrainState, span: SpanResult) -> NodeUpdate:
        existing = state.get("existing")
        symptom_text = existing.text if existing else state["normalized_symptom"]
        symptom_id, action_ids = await self._deps.knowledge.save_proposals(
            symptom_text=symptom_text,
            embedding=state["embedding"],
            existing_id=existing.symptom_id if existing else None,
            proposals=state["proposals"],
            decisions=state.get("decisions", []),
            prompt_version=state.get("prompt_version"),
        )
        span.summary = (
            f"stored {len(action_ids)} actions for "
            f"{'existing' if existing else 'new'} symptom '{symptom_text}'"
        )
        span.payload = {"symptom_id": str(symptom_id)}
        return {"saved_symptom_id": symptom_id, "saved_action_ids": action_ids}
