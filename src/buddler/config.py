"""Application settings, loaded from environment variables and `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, PostgresDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RoomConfig(BaseModel):
    """Homematic channels of one room (CCU-Jack addresses `DEVICE:CHANNEL`)."""

    # Thermostat / heating group channel: ACTUAL_TEMPERATURE, SET_POINT_TEMPERATURE, ...
    climate: str = Field(pattern=r"^[A-Za-z0-9-]+:\d+$")
    # Optional window contact channel (STATE); else WINDOW_STATE of the climate channel.
    window: str | None = Field(default=None, pattern=r"^[A-Za-z0-9-]+:\d+$")
    # Other names users may use for the room ("wohnzimmer", "lounge").
    aliases: list[str] = []


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM gateway (OpenAI-compatible, e.g. Requesty). "fake" = offline, deterministic.
    llm_provider: Literal["openai", "fake"] = "openai"
    llm_base_url: str = "https://router.requesty.ai/v1"
    llm_api_key: SecretStr = SecretStr("")
    llm_chat_model: str = "openai/gpt-4o-mini"
    llm_embed_model: str = "openai/text-embedding-3-small"
    # Fixed by the DB schema; changing it requires a migration + re-embedding.
    embed_dim: int = 1536
    # json_schema is supported by some models only; json_object puts the schema in the prompt.
    llm_structured_mode: Literal["json_schema", "json_object"] = "json_schema"
    llm_structured_retries: int = Field(default=2, ge=0)
    llm_temperature: float = Field(default=0.0, ge=0, le=2)
    llm_timeout_s: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0)
    llm_max_input_tokens: int = Field(default=16000, ge=1)
    embed_batch_size: int = Field(default=128, ge=1)
    embed_concurrency: int = Field(default=4, ge=1)
    embed_max_input_chars: int = Field(default=8000, ge=1)

    database_url: PostgresDsn = PostgresDsn(
        "postgresql+asyncpg://app_rw:app_rw@localhost:5432/buddler"
    )
    mcp_url: str = "http://localhost:8001/sse"
    mcp_timeout_s: float = Field(default=15.0, gt=0)
    mcp_max_rows: int = Field(default=50, ge=1)

    # Agent loop bounds (AGENTS.md §6)
    agent_max_steps: int = Field(default=8, ge=1)
    agent_max_tool_calls: int = Field(default=6, ge=0)
    llm_max_tokens_per_call: int = Field(default=4000, ge=1)

    # Location: local time and weather.
    timezone: str = "Europe/Berlin"
    location_latitude: float = Field(default=52.52, ge=-90, le=90)
    location_longitude: float = Field(default=13.405, ge=-180, le=180)

    # Weather (ADR 0005)
    weather_provider: Literal["open_meteo", "fake"] = "open_meteo"
    weather_base_url: str = "https://api.open-meteo.com/v1/forecast"
    weather_forecast_hours: int = Field(default=6, ge=1, le=48)
    weather_timeout_s: float = Field(default=10.0, gt=0)
    fake_weather: Literal["cloudy", "sunny"] = "cloudy"

    # Homematic via CCU-Jack (ADR 0004)
    homematic_provider: Literal["ccu_jack", "fake"] = "ccu_jack"
    homematic_endpoint: str = "https://homematic-raspi:2122"
    homematic_username: str = ""
    homematic_password: SecretStr = SecretStr("")
    homematic_verify_tls: bool = True
    homematic_timeout_s: float = Field(default=10.0, gt=0)
    # JSON in env: {"living room": {"climate": "000A1B2C3D4E5F:1", "aliases": ["wohnzimmer"]}}
    homematic_rooms: dict[str, RoomConfig] = {}
    default_room: str | None = None

    # Actuation bounds (ADR 0002). Empty room list = the agent never writes.
    actuation_rooms: list[str] = []
    setpoint_min_c: float = 16.0
    setpoint_max_c: float = 24.0
    setpoint_max_delta_c: float = Field(default=2.0, gt=0)
    actuation_max_writes_per_run: int = Field(default=1, ge=0)
    actuation_cooldown_min: int = Field(default=30, ge=0)
    dry_run: bool = False

    # Resolve workflow
    resolve_top_k: int = Field(default=3, ge=1)
    resolve_max_distance: float = Field(default=0.45, ge=0, le=2)
    # Train: cosine distance under which a new symptom counts as an existing one.
    train_duplicate_max_distance: float = Field(default=0.15, ge=0, le=2)

    # Static bearer key for the local demonstrator API (empty = all API calls rejected).
    api_key: SecretStr = SecretStr("")
    sse_poll_interval_s: float = Field(default=0.5, gt=0)
    sse_heartbeat_s: float = Field(default=15.0, gt=0)
    data_dir: Path = Path("data")

    @field_validator("timezone")
    @classmethod
    def _known_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as err:
            raise ValueError(f"unknown timezone {value!r}") from err
        return value

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


@lru_cache
def get_settings() -> Settings:
    return Settings()
