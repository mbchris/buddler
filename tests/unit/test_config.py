import pytest

from buddler.config import Settings


def test_settings_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_CHAT_MODEL", "anthropic/claude-sonnet-5-5")
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    settings = Settings(_env_file=None)

    assert settings.llm_chat_model == "anthropic/claude-sonnet-5-5"
    assert settings.llm_api_key.get_secret_value() == "test-key"


def test_secrets_are_masked_in_dump() -> None:
    settings = Settings(_env_file=None, llm_api_key="secret")

    assert "secret" not in str(settings.model_dump())


@pytest.mark.parametrize("value", ["0", "-1"])
def test_agent_max_steps_must_be_positive(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("AGENT_MAX_STEPS", value)

    with pytest.raises(ValueError, match="agent_max_steps"):
        Settings(_env_file=None)
