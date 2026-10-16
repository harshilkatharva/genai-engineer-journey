from __future__ import annotations

import os

import pytest

from genai_core.core import Settings, get_settings


@pytest.fixture
def mock_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    settings = Settings(
        app_name="Test Core",
        default_llm_provider="google",
        default_llm_model="gemini-3.5-flash-lite",
        default_llm_temperature=0.2,
        max_structured_error_retry=2,
        integration_test=0,
        google_api_key="test-google-key",
        openai_api_key="test-openai-key",
        anthropic_api_key="test-anthropic-key",
    )
    monkeypatch.setattr("genai_core.core.settings.get_settings", lambda: settings)
    monkeypatch.setattr("genai_core.core.get_settings", lambda: settings)
    return settings


@pytest.fixture(autouse=True)
def require_integration_test_flag(request: pytest.FixtureRequest) -> None:
    if "integration" in request.keywords:
        settings = get_settings()
        is_enabled = (
            getattr(settings, "integration_test", 0) == 1 or os.getenv("INTEGRATION_TEST") == "1"
        )
        if not is_enabled:
            pytest.skip(
                "Integration tests are disabled. Set INTEGRATION_TEST=1 in .env to run them."
            )
