"""Plain-language step explanations for the demonstrator view.

Thresholds are read from settings so the text always matches the running configuration.
"""

from dataclasses import dataclass

from buddler.agents.nodes.resolve import MAX_OBSERVATION_CHARS
from buddler.config import Settings
from buddler.models.enums import RunKind


@dataclass(frozen=True)
class StepExplanation:
    title: str
    # One line, used as graph tooltip and list headline.
    summary: str
    # What the step does, in words a non-specialist can follow.
    description: str
    why: str
    example: str
    # Implementation facts for readers who want the mechanics.
    details: tuple[str, ...]


@dataclass(frozen=True)
class WorkflowExplanation:
    overview: str
    glossary: tuple[tuple[str, str], ...]
    steps: dict[str, StepExplanation]


GLOSSARY = (
    (
        "Symptom",
        "How a person feels or what they want, as a short curated phrase ('feeling cold'). "
        "Users phrase it freely; matching uses meaning (embeddings), not exact words.",
    ),
    (
        "Action",
        "What to do for a symptom: advice for the person, or a heating change. Every action "
        "has conditions (time window, weather, room temperature, window) checked by code.",
    ),
    (
        "S / A / O refs",
        "Short ids in the AI's context: S1 = matched symptom, A1 = candidate action, "
        "O1 time, O2 weather, O3 room climate. The AI must cite them; code maps them back and "
        "flags uncited claims.",
    ),
    (
        "Actuation guard",
        "Deterministic gate before every device write: allowed rooms, setpoint range, max "
        "change, writes per run, cooldown, dry run. The AI cannot bypass it.",
    ),
)

RESOLVE_OVERVIEW = (
    "Linear LangGraph pipeline with nine nodes. Two short AI calls frame and decide; "
    "everything in between is observation and rule checking done by code, plus one bounded "
    "AI investigation loop. The AI only picks among knowledge-backed candidates; device "
    "writes pass a deterministic guard and are logged."
)


def resolve_explanation(settings: Settings) -> WorkflowExplanation:
    rooms = ", ".join(settings.actuation_rooms) or "none (writes disabled)"
    steps = {
        "understand": StepExplanation(
            title="Understand",
            summary="Turn the sentence into a symptom phrase and an optional room.",
            description=(
                "One AI call reads the sentence and returns a short, normalized symptom "
                "('feeling cold'), the room the person mentions, and an urgency."
            ),
            why="Knowledge is curated per symptom; free wording has to be mapped onto it.",
            example="'Mir ist kalt im Wohnzimmer' -> symptom 'feeling cold', room 'Wohnzimmer'.",
            details=(
                "Structured output validated against a schema; invalid output is retried.",
                "The sentence is wrapped as untrusted data; on failure the raw sentence is used.",
            ),
        ),
        "locate": StepExplanation(
            title="Locate",
            summary="Decide which room the person is in.",
            description=(
                "Explicit room argument first, then a room named in the sentence (names and "
                "aliases from the configuration), then the default room."
            ),
            why="Temperature and actions are per room; the wrong room means the wrong action.",
            example="'Wohnzimmer' matches the alias of 'living room'.",
            details=(f"Default room: {settings.default_room or 'none'}.",),
        ),
        "observe": StepExplanation(
            title="Observe",
            summary="Read the clock, the weather and the room climate.",
            description=(
                "Local time (or a simulated time for demos), outdoor temperature and the "
                "sunshine forecast, and the room's temperature, setpoint, humidity and window "
                "state from Homematic."
            ),
            why="Whether heating makes sense depends on facts, not on the AI's guess.",
            example="02:00, 4 C outside, no sun; living room 19.5 C, setpoint 20 C.",
            details=(
                f"Weather: {settings.weather_provider}, next {settings.weather_forecast_hours} h.",
                f"Homematic: {settings.homematic_provider}.",
                "A failing source becomes 'unknown' (traced), the run continues.",
            ),
        ),
        "retrieve": StepExplanation(
            title="Retrieve knowledge",
            summary="Find curated symptoms similar to this one.",
            description=(
                "The symptom phrase is embedded and compared with confirmed symptoms in the "
                "knowledge base; the closest ones bring their confirmed actions."
            ),
            why="Developer-confirmed knowledge, not the AI's general knowledge, drives actions.",
            example="'freezing' is close to the curated 'feeling cold'.",
            details=(
                f"Top {settings.resolve_top_k}, cosine distance <= "
                f"{settings.resolve_max_distance}; pgvector HNSW index.",
            ),
        ),
        "evaluate": StepExplanation(
            title="Evaluate conditions",
            summary="Check every candidate action's conditions against the observations.",
            description=(
                "Each condition is met, violated or unknown. Advice is usable unless a "
                "condition is violated; heating actions additionally must pass the actuation "
                "guard (all conditions met, room allowed, within limits)."
            ),
            why="Reproducible rules: the same observations always give the same candidates.",
            example="At 02:00 'raise heating 1 C' violates its 06:00-22:00 window.",
            details=(
                f"Writable rooms: {rooms}.",
                f"Setpoint {settings.setpoint_min_c:g}-{settings.setpoint_max_c:g} C, max change "
                f"{settings.setpoint_max_delta_c:g} C.",
            ),
        ),
        "agent": StepExplanation(
            title="Investigate (agent)",
            summary="Bounded AI loop: check history and other rooms before deciding.",
            description=(
                "The AI may query the read-only knowledge views (e.g. recent heating changes) "
                "and read other rooms, then writes short notes."
            ),
            why="Avoids escalating (heating raised twice) and finds explanations (open window).",
            example="Query: last device commands for the living room -> none in the last hour.",
            details=(
                f"Max {settings.agent_max_steps} steps, {settings.agent_max_tool_calls} tool "
                "calls; SQL passes a guard (single SELECT, row cap) before the MCP server.",
                f"Tool output is truncated to {MAX_OBSERVATION_CHARS} characters.",
            ),
        ),
        "decide": StepExplanation(
            title="Decide",
            summary="AI picks candidate actions by ref and writes the reply.",
            description=(
                "The AI chooses among the evaluated candidates, cites refs, adds advice and "
                "writes a short reply. It cannot invent device actions."
            ),
            why="Language and judgment from the AI, facts and limits from code.",
            example="Chosen A3 'tea or sweater' citing O1 (night) and A1 (heating not allowed).",
            details=("Unknown refs are dropped and listed; uncited claims are flagged.",),
        ),
        "act": StepExplanation(
            title="Act",
            summary="Apply chosen heating changes through the actuation guard.",
            description=(
                "Each chosen device action is checked again (writes per run, cooldown) and "
                "then written to Homematic, or only logged in dry-run mode."
            ),
            why="The final, deterministic safety net in front of the real home.",
            example="Living room setpoint 20 -> 21 C (applied).",
            details=(
                f"Max {settings.actuation_max_writes_per_run} write(s) per run, cooldown "
                f"{settings.actuation_cooldown_min} min, dry run {settings.dry_run}.",
                "Every outcome is stored in device_command.",
            ),
        ),
        "report": StepExplanation(
            title="Report",
            summary="Store the resolution.",
            description="Reply, actions with outcomes, advice and observations are persisted.",
            why="The trace view and CLI show results without re-running the workflow.",
            example="resolution stored for run 3f2c…",
            details=("Table app.resolution (JSONB body + citations).",),
        ),
    }
    return WorkflowExplanation(overview=RESOLVE_OVERVIEW, glossary=GLOSSARY, steps=steps)


TRAIN_OVERVIEW = (
    "Linear LangGraph pipeline with five nodes. The AI proposes actions for a symptom; a "
    "developer reviews them. Only confirmed knowledge is used by the resolve workflow."
)


def train_explanation(settings: Settings) -> WorkflowExplanation:
    steps = {
        "embed": StepExplanation(
            title="Embed",
            summary="Turn the symptom phrase into a vector.",
            description="The phrase is embedded so it can be compared by meaning.",
            why="Finds duplicates even when worded differently.",
            example="'feeling chilly' -> 1536 numbers.",
            details=("Embeddings are cached per model and text hash.",),
        ),
        "match": StepExplanation(
            title="Match existing",
            summary="Look for a symptom that already exists.",
            description="The closest symptom of any status is loaded with its actions.",
            why="New actions attach to the existing symptom instead of duplicating it.",
            example="'feeling chilly' matches 'feeling cold' (distance 0.08).",
            details=(f"Duplicate if distance <= {settings.train_duplicate_max_distance}.",),
        ),
        "propose": StepExplanation(
            title="Propose",
            summary="AI proposes actions with structured conditions.",
            description=(
                "The AI suggests advice and heating actions with conditions, complementing "
                "existing actions. Device actions outside the configured limits are dropped."
            ),
            why="Speeds up curation; a human still decides.",
            example="'raise heating 1 C' only 06:00-22:00, window closed, sun <= 1 h.",
            details=("Structured output; unknown condition fields are rejected.",),
        ),
        "review": StepExplanation(
            title="Review",
            summary="Developer confirms, rejects or defers each proposal.",
            description="Interactive in the CLI, or auto-confirm / defer for later review.",
            why="Developer-confirmed knowledge outranks AI proposals.",
            example="Confirm 2, reject 1.",
            details=("Interactive review pauses the graph (LangGraph interrupt).",),
        ),
        "persist": StepExplanation(
            title="Persist",
            summary="Store the symptom and actions with provenance.",
            description="Existing entries are never overwritten; new actions are added.",
            why="Keeps human-confirmed knowledge intact.",
            example="2 confirmed actions stored for 'feeling cold'.",
            details=("Provenance 'llm' and the prompt version are stored per action.",),
        ),
    }
    return WorkflowExplanation(overview=TRAIN_OVERVIEW, glossary=GLOSSARY, steps=steps)


def explanation(kind: RunKind, settings: Settings) -> WorkflowExplanation:
    if kind is RunKind.TRAIN:
        return train_explanation(settings)
    return resolve_explanation(settings)
