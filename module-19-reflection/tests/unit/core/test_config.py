"""Unit tests for runtime configuration (ReportWriterConfig)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from self_improving_report_writer.core.config import (
    ReportWriterConfig,
    get_default_config,
)


def test_default_config() -> None:
    """Test default values of ReportWriterConfig."""
    config = get_default_config()
    assert config.generator_provider == "google"
    assert config.generator_model == "gemini-3.5-flash-lite"
    assert config.judge_provider == "google"
    assert config.judge_model == "gemini-3.1-flash-lite"
    assert config.default_top_k == 8
    assert config.min_retrieval_score == 0.0
    assert config.max_reflection_rounds == 3
    assert config.max_total_tokens == 50000
    assert config.max_total_cost_usd == 0.20
    assert config.max_wall_clock_seconds == 90.0
    assert config.acceptance_threshold == 0.85
    assert config.batch_claim_evaluation is True
    assert config.max_claims_per_batch == 6
    assert config.eval_rate_limit_max_retries == 3
    assert config.eval_rate_limit_safety_buffer == 1.0
    assert config.eval_rate_limit_default_backoff == 10.0


def test_config_boundary_validation() -> None:
    """Test validation constraints on ReportWriterConfig fields."""
    # default_top_k must be between 1 and 50
    with pytest.raises(ValidationError):
        ReportWriterConfig(default_top_k=0)
    with pytest.raises(ValidationError):
        ReportWriterConfig(default_top_k=51)

    # min_retrieval_score must be between 0.0 and 1.0
    with pytest.raises(ValidationError):
        ReportWriterConfig(min_retrieval_score=-0.1)
    with pytest.raises(ValidationError):
        ReportWriterConfig(min_retrieval_score=1.1)

    # max_reflection_rounds must be between 1 and 10
    with pytest.raises(ValidationError):
        ReportWriterConfig(max_reflection_rounds=0)
    with pytest.raises(ValidationError):
        ReportWriterConfig(max_reflection_rounds=11)

    # max_total_tokens must be >= 1000
    with pytest.raises(ValidationError):
        ReportWriterConfig(max_total_tokens=999)

    # max_wall_clock_seconds must be >= 5.0
    with pytest.raises(ValidationError):
        ReportWriterConfig(max_wall_clock_seconds=4.9)

    # max_claims_per_batch must be between 1 and 20
    with pytest.raises(ValidationError):
        ReportWriterConfig(max_claims_per_batch=0)
    with pytest.raises(ValidationError):
        ReportWriterConfig(max_claims_per_batch=21)
