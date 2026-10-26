"""Domain enums shared by models, schemas and services."""

from enum import StrEnum


class Provenance(StrEnum):
    HUMAN = "human"
    LLM = "llm"


class KnowledgeStatus(StrEnum):
    PROPOSED = "proposed"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class ActionKind(StrEnum):
    # Text for the user; never touches a device.
    ADVICE = "advice"
    # Relative change of the room's thermostat setpoint (params.delta_c).
    HEATING_DELTA = "heating_delta"
    # Absolute thermostat setpoint (params.setpoint_c).
    HEATING_SETPOINT = "heating_setpoint"


DEVICE_ACTION_KINDS = frozenset({ActionKind.HEATING_DELTA, ActionKind.HEATING_SETPOINT})


class CommandStatus(StrEnum):
    APPLIED = "applied"
    DRY_RUN = "dry_run"
    REJECTED = "rejected"
    FAILED = "failed"


class RunKind(StrEnum):
    TRAIN = "train"
    RESOLVE = "resolve"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_REVIEW = "waiting_review"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class TraceEventType(StrEnum):
    NODE_START = "node_start"
    NODE_END = "node_end"
    LLM_CALL = "llm_call"
    EMBEDDING_CALL = "embedding_call"
    TOOL_CALL = "tool_call"
    MCP_CALL = "mcp_call"
    HOME_CALL = "home_call"
    WEATHER_CALL = "weather_call"
    ERROR = "error"
