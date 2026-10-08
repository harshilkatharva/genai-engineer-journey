from __future__ import annotations

import pytest
from pydantic import ValidationError

from memory_system.core.config import get_settings
from tests.fakes import make_settings


def test_settings_read_validation_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings(app_name="test app", memory_token_budget=120)
    assert settings.app_name == "test app"
    assert settings.default_llm_provider == "google"
    assert settings.memory_token_budget == 120
    assert settings.database_url == "postgresql://localhost:5432/memory"

    values = [
        (
            "DATABASE_CONNECTION_CONVERSATION_URL",
            "postgresql://primary/db",
            "database_url",
            "postgresql://primary/db",
        ),
        ("DATABASE_URL", "postgresql://fallback/db", "database_url", "postgresql://fallback/db"),
        ("MEMORY_EMBEDDING_DIMENSION", "12", "memory_embedding_dimension", 12),
        ("INTEGRATION_TEST", "true", "integration_test", True),
    ]
    for name, value, attribute, expected in values:
        monkeypatch.setenv(name, value)
        assert getattr(make_settings(), attribute) == expected
        monkeypatch.delenv(name)


def test_settings_reject_invalid_memory_limits() -> None:
    invalid_values = {
        "memory_token_budget": 0,
        "memory_working_ttl_hours": 0,
        "memory_episodic_retention_days": 0,
        "memory_embedding_dimension": 0,
        "memory_retrieval_limit": 0,
        "memory_recency_half_life_days": 0,
        "memory_embedding_model": "",
    }
    for field, value in invalid_values.items():
        with pytest.raises(ValidationError):
            make_settings(**{field: value})


def test_get_settings_caches_and_can_be_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("MEMORY_TOKEN_BUDGET", "321")

    first = get_settings()
    monkeypatch.setenv("MEMORY_TOKEN_BUDGET", "654")
    cached = get_settings()
    get_settings.cache_clear()
    refreshed = get_settings()

    assert first is cached
    assert cached.memory_token_budget == 321
    assert refreshed.memory_token_budget == 654
    get_settings.cache_clear()
