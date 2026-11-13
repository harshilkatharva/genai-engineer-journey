"""Unit tests for rubric domain models (DimensionScore, ReportJudgeVerdict, SemanticGroundingReport)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from self_improving_report_writer.models.rubric import (
    ClaimGroundingResult,
    DimensionScore,
    ReportJudgeVerdict,
    SemanticGroundingReport,
)
from self_improving_report_writer.models.verification import (
    DefectCategory,
    DefectSeverity,
    VerificationIssue,
)


def test_dimension_score_valid() -> None:
    """Test valid DimensionScore creation and attributes."""
    ds = DimensionScore(
        dimension_name="factual_accuracy",
        score=0.92,
        passed=True,
        reasoning="All factual assertions are substantiated.",
    )
    assert ds.dimension_name == "factual_accuracy"
    assert ds.score == 0.92
    assert ds.passed is True
    assert "substantiated" in ds.reasoning


def test_dimension_score_boundary_validation() -> None:
    """Test that score is strictly bounded between 0.0 and 1.0."""
    # Score 0.0 and 1.0 are valid
    ds_min = DimensionScore(
        dimension_name="relevance", score=0.0, passed=False, reasoning="Low relevance"
    )
    assert ds_min.score == 0.0

    ds_max = DimensionScore(
        dimension_name="relevance", score=1.0, passed=True, reasoning="High relevance"
    )
    assert ds_max.score == 1.0

    with pytest.raises(ValidationError):
        DimensionScore(
            dimension_name="relevance", score=-0.01, passed=False, reasoning="Below zero"
        )

    with pytest.raises(ValidationError):
        DimensionScore(dimension_name="relevance", score=1.01, passed=True, reasoning="Above one")


def test_report_judge_verdict_valid() -> None:
    """Test valid ReportJudgeVerdict creation and default values."""
    ds = DimensionScore(
        dimension_name="completeness",
        score=0.88,
        passed=True,
        reasoning="Good coverage of topic.",
    )
    verdict = ReportJudgeVerdict(
        composite_score=0.88,
        passed=True,
        dimension_scores={"completeness": ds},
        overall_feedback="Comprehensive report with minor omissions.",
    )
    assert verdict.composite_score == 0.88
    assert verdict.passed is True
    assert "completeness" in verdict.dimension_scores
    assert verdict.is_self_judged is False
    assert verdict.judge_model == "gemini-3.1-flash-lite"
    assert verdict.tokens_used == 0
    assert verdict.cost_usd == 0.0


def test_report_judge_verdict_bounds() -> None:
    """Test composite_score validation boundaries."""
    with pytest.raises(ValidationError):
        ReportJudgeVerdict(
            composite_score=1.05,
            passed=True,
            dimension_scores={},
            overall_feedback="Invalid score",
        )

    with pytest.raises(ValidationError):
        ReportJudgeVerdict(
            composite_score=-0.1,
            passed=False,
            dimension_scores={},
            overall_feedback="Invalid score",
        )


def test_claim_grounding_result_valid() -> None:
    """Test valid ClaimGroundingResult with literals ENTAILS, NEUTRAL, CONTRADICTS."""
    res_entails = ClaimGroundingResult(
        claim_id="c1",
        section_id="sec1",
        entailment="ENTAILS",
        score=1.0,
        reasoning="Clearly entailed",
        cited_chunk_ids=["chunk_1"],
    )
    assert res_entails.entailment == "ENTAILS"
    assert res_entails.score == 1.0

    res_neutral = ClaimGroundingResult(
        claim_id="c2",
        section_id="sec1",
        entailment="NEUTRAL",
        score=0.5,
        reasoning="Insufficient evidence",
    )
    assert res_neutral.entailment == "NEUTRAL"
    assert res_neutral.cited_chunk_ids == []

    res_contra = ClaimGroundingResult(
        claim_id="c3",
        section_id="sec1",
        entailment="CONTRADICTS",
        score=0.0,
        reasoning="Source contradicts statement",
    )
    assert res_contra.entailment == "CONTRADICTS"

    with pytest.raises(ValidationError):
        ClaimGroundingResult(
            claim_id="c4",
            section_id="sec1",
            entailment="INVALID_ENUM",  # type: ignore[arg-type]
            score=0.5,
            reasoning="Invalid enum",
        )


def test_semantic_grounding_report_valid() -> None:
    """Test SemanticGroundingReport creation and defect counts."""
    issue = VerificationIssue(
        issue_id="ISSUE-G01",
        category=DefectCategory.FACTUAL_GROUNDING,
        severity=DefectSeverity.MAJOR,
        message="Claim not supported by cited source.",
    )
    rep = SemanticGroundingReport(
        passed=False,
        all_claims_grounded=False,
        grounding_score=0.75,
        claim_evaluations=[],
        issues=[issue],
        critical_count=0,
        major_count=1,
        minor_count=0,
        tokens_used=1200,
        cost_usd=0.012,
    )
    assert rep.passed is False
    assert rep.grounding_score == 0.75
    assert len(rep.issues) == 1
    assert rep.major_count == 1
    assert rep.tokens_used == 1200
    assert rep.cost_usd == 0.012


def test_semantic_grounding_report_bounds() -> None:
    """Test boundary checks on SemanticGroundingReport."""
    with pytest.raises(ValidationError):
        SemanticGroundingReport(
            passed=True,
            all_claims_grounded=True,
            grounding_score=1.1,  # > 1.0 invalid
        )

    with pytest.raises(ValidationError):
        SemanticGroundingReport(
            passed=True,
            all_claims_grounded=True,
            grounding_score=-0.1,  # < 0.0 invalid
        )
