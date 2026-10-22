"""Unit tests for agent runtime configuration."""

from __future__ import annotations

import os
from unittest.mock import patch

from research_and_support_agent.core.config import AgentConfig, get_agent_config


def test_default_config_limits() -> None:
    """Verify default boundary limits comply with business logic specifications."""
    cfg = AgentConfig()
    assert cfg.default_max_steps == 10
    assert cfg.default_max_tokens == 10_000
    assert cfg.default_max_cost_usd == 0.05
    assert cfg.default_timeout_seconds == 30.0
    assert cfg.default_max_repeated_calls == 3


def test_default_web_search_settings() -> None:
    """Verify default web search constraints."""
    cfg = AgentConfig()
    assert cfg.web_search_max_results == 3
    assert cfg.web_search_timeout_seconds == 10.0
    assert cfg.web_search_max_chars == 4000


def test_environment_variable_overrides() -> None:
    """Verify configuration loads environment variables properly."""
    with patch.dict(
        os.environ,
        {
            "DEFAULT_LLM_MODEL": "gemini-3.8-flash",
            "DEFAULT_LLM_PROVIDER": "google_custom",
            "DATABASE_CONNECTION_CONVERSATION_URL": "postgresql://test:test@127.0.0.1:5432/test_db",
        },
    ):
        cfg = AgentConfig()
        assert cfg.default_model == "gemini-3.8-flash"
        assert cfg.default_provider == "google_custom"
        assert cfg.postgres_url == "postgresql://test:test@127.0.0.1:5432/test_db"


def test_get_agent_config_singleton() -> None:
    """Verify factory helper returns an AgentConfig instance."""
    cfg = get_agent_config()
    assert isinstance(cfg, AgentConfig)
