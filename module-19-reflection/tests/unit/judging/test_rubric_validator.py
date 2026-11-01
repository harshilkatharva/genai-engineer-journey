"""Unit tests for RubricConfig and RubricCriterion validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from self_improving_report_writer.models.rubric import (
    RubricConfig,
    RubricCriterion,
    get_default_rubric_config,
)


def test_default_rubric_config_valid() -> None:
    """Verify default 7-dimension rubric is valid and sums to 1.0."""
    config = get_default_rubric_config()
    assert len(config.criteria) == 7
    total_weight = sum(c.weight for c in config.criteria)
    assert abs(total_weight - 1.0) < 1e-4
    assert config.acceptance_threshold == 0.85

    expected_names = {
        "factual_accuracy",
        "evidence_grounding",
        "citation_correctness",
        "completeness",
        "relevance",
        "clarity_organization",
        "logical_consistency",
    }
    actual_names = {c.name for c in config.criteria}
    assert actual_names == expected_names


def test_weights_must_sum_to_one() -> None:
    """Verify criteria whose weights do not sum to 1.0 are rejected."""
    # Under-weighted
    with pytest.raises(ValidationError, match="Rubric criteria weights must sum to 1.0"):
        RubricConfig(
            criteria=[
                RubricCriterion(name="accuracy", weight=0.5, description="Check accuracy"),
                RubricCriterion(name="clarity", weight=0.3, description="Check clarity"),
            ]
        )

    # Over-weighted
    with pytest.raises(ValidationError, match="Rubric criteria weights must sum to 1.0"):
        RubricConfig(
            criteria=[
                RubricCriterion(name="accuracy", weight=0.7, description="Check accuracy"),
                RubricCriterion(name="clarity", weight=0.5, description="Check clarity"),
            ]
        )


def test_empty_criteria_rejected() -> None:
    """Verify rubric with no criteria is rejected."""
    with pytest.raises(ValidationError, match="Rubric must contain at least one criterion"):
        RubricConfig(criteria=[])


def test_duplicate_criterion_rejected() -> None:
    """Verify duplicate criterion names are rejected even with different casing."""
    with pytest.raises(ValidationError, match="Duplicate criterion name detected"):
        RubricConfig(
            criteria=[
                RubricCriterion(name="Accuracy", weight=0.5, description="Check accuracy 1"),
                RubricCriterion(name="accuracy", weight=0.5, description="Check accuracy 2"),
            ]
        )


def test_empty_criterion_name_or_description_rejected() -> None:
    """Verify whitespace-only or empty name and description are rejected."""
    with pytest.raises(ValidationError, match="Field cannot be empty"):
        RubricCriterion(name="   ", weight=0.5, description="Valid description")

    with pytest.raises(ValidationError, match="Field cannot be empty"):
        RubricCriterion(name="Valid Name", weight=0.5, description="  ")


def test_criterion_bounds() -> None:
    """Verify weight and min_pass_score bounds."""
    with pytest.raises(ValidationError):
        RubricCriterion(name="test", weight=-0.1, description="desc")

    with pytest.raises(ValidationError):
        RubricCriterion(name="test", weight=1.1, description="desc")

    with pytest.raises(ValidationError):
        RubricCriterion(name="test", weight=0.5, description="desc", min_pass_score=1.5)
