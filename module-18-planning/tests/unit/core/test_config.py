"""Unit tests for AssistantConfig and core configuration."""

from __future__ import annotations

import os
from unittest.mock import patch

from plan_and_execute_research_assistant.core import AssistantConfig, get_assistant_config


def test_assistant_config_defaults() -> None:
    """Verify default configuration parameters."""
    config = AssistantConfig()
    assert config.default_model == "gemini-3.5-flash-lite"
    assert config.default_provider is None
    assert config.default_temperature == 0.2
    assert config.max_planning_retries == 3
    assert config.max_concurrency == 3
    assert config.auto_approve_review is False
    assert config.max_step_retries == 2
    assert config.max_replans == 3
    assert config.max_total_tokens == 40_000
    assert config.max_total_cost_usd == 0.25
    assert config.default_timeout_seconds == 180.0
    assert config.verification_threshold == 0.85
    # Config-based gate: episodic similarity search disabled by default
    assert config.enable_episodic_similarity_search is False
    assert config.max_recalled_preferences == 5
    assert config.max_recalled_past_plans == 3


def test_assistant_config_env_override() -> None:
    """Verify environment variables override defaults."""
    env = {
        "ASSISTANT_ENABLE_EPISODIC_SIMILARITY_SEARCH": "true",
        "ASSISTANT_DEFAULT_MODEL": "gemini-3.8-flash",
        "ASSISTANT_MAX_PLANNING_RETRIES": "5",
        "ASSISTANT_AUTO_APPROVE_REVIEW": "true",
    }
    with patch.dict(os.environ, env):
        config = AssistantConfig()
        assert config.enable_episodic_similarity_search is True
        assert config.default_model == "gemini-3.8-flash"
        assert config.max_planning_retries == 5
        assert config.auto_approve_review is True


def test_get_assistant_config_singleton() -> None:
    """Verify get_assistant_config returns a valid cached instance."""
    cfg1 = get_assistant_config()
    cfg2 = get_assistant_config()
    assert cfg1 is cfg2
