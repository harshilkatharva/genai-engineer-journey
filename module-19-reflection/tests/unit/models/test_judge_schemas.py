"""Unit tests for judge prompt response schemas in judging/prompts.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from self_improving_report_writer.judging.prompts import (
    ClaimEntailmentEvaluation,
    LLMJudgeEvaluation,
    ProseConsistencyEvaluation,
    SectionClaimsEntailmentEvaluation,
    SingleClaimEntailmentVerdict,
)
from self_improving_report_writer.models.rubric import DimensionScore


def test_llm_judge_evaluation_schema() -> None:
    """Test valid LLMJudgeEvaluation instantiation and validation."""
    ds1 = DimensionScore(
        dimension_name="factual_accuracy",
        score=0.90,
        passed=True,
        reasoning="Good facts",
    )
    eval_resp = LLMJudgeEvaluation(
        dimension_scores=[ds1],
        overall_feedback="Well researched report with solid grounding.",
    )
    assert len(eval_resp.dimension_scores) == 1
    assert eval_resp.dimension_scores[0].dimension_name == "factual_accuracy"
    assert "solid grounding" in eval_resp.overall_feedback


def test_claim_entailment_evaluation_schema() -> None:
    """Test ClaimEntailmentEvaluation schema validation."""
    ev = ClaimEntailmentEvaluation(
        entailment="ENTAILS",
        score=1.0,
        reasoning="The source specifically confirms this finding.",
    )
    assert ev.entailment == "ENTAILS"
    assert ev.score == 1.0

    # Bounds check
    with pytest.raises(ValidationError):
        ClaimEntailmentEvaluation(
            entailment="ENTAILS",
            score=1.2,
            reasoning="Invalid score",
        )

    # Enum check
    with pytest.raises(ValidationError):
        ClaimEntailmentEvaluation(
            entailment="MAYBE",  # type: ignore[arg-type]
            score=0.5,
            reasoning="Invalid entailment",
        )


def test_single_claim_entailment_verdict_schema() -> None:
    """Test SingleClaimEntailmentVerdict schema."""
    verdict = SingleClaimEntailmentVerdict(
        claim_id="claim_001",
        entailment="NEUTRAL",
        score=0.5,
        reasoning="Passage discusses related technology but does not assert 3x speedup.",
    )
    assert verdict.claim_id == "claim_001"
    assert verdict.entailment == "NEUTRAL"
    assert verdict.score == 0.5


def test_section_claims_entailment_evaluation_schema() -> None:
    """Test SectionClaimsEntailmentEvaluation schema with multiple verdicts."""
    v1 = SingleClaimEntailmentVerdict(
        claim_id="c1", entailment="ENTAILS", score=1.0, reasoning="ok"
    )
    v2 = SingleClaimEntailmentVerdict(
        claim_id="c2", entailment="CONTRADICTS", score=0.0, reasoning="refuted"
    )
    batch = SectionClaimsEntailmentEvaluation(evaluations=[v1, v2])
    assert len(batch.evaluations) == 2
    assert batch.evaluations[0].claim_id == "c1"
    assert batch.evaluations[1].claim_id == "c2"


def test_prose_consistency_evaluation_schema() -> None:
    """Test ProseConsistencyEvaluation schema."""
    pce_empty = ProseConsistencyEvaluation(
        unstructured_claims=[],
        reasoning="All prose assertions are fully covered by structured claims.",
    )
    assert pce_empty.unstructured_claims == []

    pce_issues = ProseConsistencyEvaluation(
        unstructured_claims=["Quantum annealing demonstrated 5x speedup in production."],
        reasoning="Found ungrounded factual statement missing from claims list.",
    )
    assert len(pce_issues.unstructured_claims) == 1
