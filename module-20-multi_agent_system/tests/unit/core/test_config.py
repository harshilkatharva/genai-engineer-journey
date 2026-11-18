"""Unit tests for ResearchTeamConfig."""

from __future__ import annotations

import os
from unittest.mock import patch

from parallel_research_team.core.config import ResearchTeamConfig, get_research_team_config


def test_default_config_values() -> None:
    """Verify default parameters align with confirmed business rules."""
    cfg = ResearchTeamConfig()

    assert cfg.default_provider == "google"
    assert cfg.default_model == "gemini-3.5-flash-lite"
    assert cfg.synthesis_model == "gemini-3.5-flash-lite"

    # Reflection & Verification Judge parameters
    assert cfg.judge_provider == "google"
    assert cfg.judge_model == "gemini-3.1-flash-lite"
    assert cfg.enable_semantic_verification is True
    assert cfg.semantic_verification_threshold == 0.70

    # Concurrency and execution limits
    assert cfg.max_concurrency == 3
    assert cfg.max_workers == 4
    assert cfg.per_worker_timeout_seconds == 45.0
    assert cfg.per_worker_max_steps == 5

    # Budget
    assert cfg.max_total_tokens == 60_000
    assert cfg.max_total_cost_usd == 0.50
    assert cfg.max_wall_clock_seconds == 180.0

    # Retrieval
    assert cfg.retrieval_mode == "in_memory"
    assert cfg.embedding_dimensions == 768


def test_env_override() -> None:
    """Verify configuration parameters can be overridden via environment variables."""
    with patch.dict(
        os.environ,
        {
            "MAX_CONCURRENCY": "5",
            "JUDGE_MODEL": "gemini-1.5-pro",
            "SEMANTIC_VERIFICATION_THRESHOLD": "0.75",
            "RETRIEVAL_MODE": "postgres",
        },
    ):
        cfg = ResearchTeamConfig()
        assert cfg.max_concurrency == 5
        assert cfg.judge_model == "gemini-1.5-pro"
        assert cfg.semantic_verification_threshold == 0.75
        assert cfg.retrieval_mode == "postgres"


def test_singleton_getter() -> None:
    """Verify get_research_team_config returns a cached instance."""
    cfg1 = get_research_team_config()
    cfg2 = get_research_team_config()
    assert cfg1 is cfg2
