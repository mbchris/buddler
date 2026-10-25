"""LLM output contracts (validated structured output, AGENTS.md §6)."""

from typing import Literal

from pydantic import BaseModel, Field

from buddler.knowledge.conditions import ActionConditions, ActionParams
from buddler.models.enums import ActionKind


class UnderstandingDraft(BaseModel):
    symptom: str = Field(
        description="Short English phrase for the user's state, e.g. 'feeling cold'."
    )
    room_mention: str | None = Field(
        default=None, description="Room the user names, verbatim; null if none."
    )
    urgency: Literal["low", "normal", "high"]


class AgentStep(BaseModel):
    thought: str = Field(description="One or two sentences: what you want to find out and why.")
    action: Literal["query", "read_room", "finish"]
    sql: str | None = Field(
        default=None, description="Single read-only SELECT on the knowledge schema (query)."
    )
    room: str | None = Field(default=None, description="Configured room name (read_room).")
    notes: str | None = Field(default=None, description="Key observations with citations (finish).")


class ChosenActionDraft(BaseModel):
    ref: str = Field(description="Candidate action ref, e.g. A2.")
    rationale: str
    citations: list[str] = Field(description="Refs like S1, A2, O1 from the context.")


class AdviceDraft(BaseModel):
    text: str
    citations: list[str] = Field(description="Refs like S1, A2, O1 from the context.")


class DecisionDraft(BaseModel):
    reply: str = Field(description="Two sentences at most, addressed to the user.")
    summary: str = Field(description="One sentence for the operator: what was decided and why.")
    chosen: list[ChosenActionDraft]
    advice: list[AdviceDraft]
    confidence: float = Field(ge=0, le=1)


class ActionProposalDraft(BaseModel):
    kind: ActionKind
    title: str = Field(description="At most 8 words.")
    instruction: str = Field(description="Advice text for the user, or what the action does.")
    params: ActionParams
    conditions: ActionConditions
    priority: int = Field(ge=1, le=1000, description="Lower = preferred.")


class ProposalResponse(BaseModel):
    symptom: str = Field(description="Normalized short English phrase for the symptom.")
    actions: list[ActionProposalDraft]
