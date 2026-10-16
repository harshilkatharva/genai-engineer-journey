import pytest

from genai_core.core import Settings


@pytest.fixture
def mock_settings(monkeypatch):
    settings = Settings(
        app_name="Test Core",
        default_llm_provider="google",
        default_llm_model="gemini-3.5-flash-lite",
        default_llm_temperature=0.2,
        max_structured_error_retry=2,
        google_api_key="test-google-key",
        openai_api_key="test-openai-key",
        anthropic_api_key="test-anthropic-key",
    )
    monkeypatch.setattr("genai_core.core.settings.get_settings", lambda: settings)
    monkeypatch.setattr("genai_core.core.get_settings", lambda: settings)
    return settings
