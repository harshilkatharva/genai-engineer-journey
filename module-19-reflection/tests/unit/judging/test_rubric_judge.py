"""Unit tests for RubricJudge: scoring, model diversity, fallback, and thresholds."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from genai_core import LLMError, LLMManagerResponse

from self_improving_report_writer.core.exceptions import JudgingError
from self_improving_report_writer.judging.prompts import LLMJudgeEvaluation
from self_improving_report_writer.judging.rubric_judge import RubricJudge
from self_improving_report_writer.models.report import Report, SourceReference
from self_improving_report_writer.models.rubric import DimensionScore


def _create_mock_eval(
    scores_dict: dict[str, float], feedback: str = "Good report"
) -> LLMJudgeEvaluation:
    """Helper to create LLMJudgeEvaluation for default criteria."""
    dim_scores = [
        DimensionScore(
            dimension_name=name,
            score=score,
            passed=score >= 0.80,
            reasoning=f"Reasoning for {name}: {score}",
        )
        for name, score in scores_dict.items()
    ]
    return LLMJudgeEvaluation(dimension_scores=dim_scores, overall_feedback=feedback)


@pytest.mark.asyncio
async def test_rubric_judge_successful_evaluation(
    valid_report: Report, sample_sources: list[SourceReference]
) -> None:
    """Verify standard rubric evaluation with distinct judge model."""
    sources_dict = {s.source_id: s for s in sample_sources}
    mock_llm = AsyncMock()

    scores = {
        "factual_accuracy": 0.95,
        "evidence_grounding": 0.90,
        "citation_correctness": 0.95,
        "completeness": 0.85,
        "relevance": 0.90,
        "clarity_organization": 0.85,
        "logical_consistency": 0.90,
    }
    raw_eval = _create_mock_eval(scores)

    mock_llm.complete_structure.return_value = LLMManagerResponse(
        structured_response=raw_eval,
        usage={"input_tokens": 500, "output_tokens": 200, "total_tokens": 700},
        raw_response={"cost": 0.0025},
    )

    judge = RubricJudge(
        llm_service=mock_llm,
        judge_model="gemini-2.5-pro",
        generator_model="gemini-3.5-flash-lite",
    )

    verdict = await judge.evaluate(valid_report, sources_dict)

    assert verdict.passed is True
    assert verdict.is_self_judged is False
    assert verdict.judge_model == "gemini-2.5-pro"
    assert verdict.tokens_used == 700
    assert verdict.cost_usd == 0.0025
    assert verdict.composite_score >= 0.90
    assert len(verdict.dimension_scores) == 7


@pytest.mark.asyncio
async def test_rubric_judge_failing_composite_score(
    valid_report: Report, sample_sources: list[SourceReference]
) -> None:
    """Verify report fails when composite score is below threshold."""
    sources_dict = {s.source_id: s for s in sample_sources}
    mock_llm = AsyncMock()

    # Scores averaging around 0.60
    scores = {
        "factual_accuracy": 0.60,
        "evidence_grounding": 0.60,
        "citation_correctness": 0.60,
        "completeness": 0.60,
        "relevance": 0.60,
        "clarity_organization": 0.60,
        "logical_consistency": 0.60,
    }
    raw_eval = _create_mock_eval(scores)

    mock_llm.complete_structure.return_value = LLMManagerResponse(
        structured_response=raw_eval,
    )

    judge = RubricJudge(llm_service=mock_llm)
    verdict = await judge.evaluate(valid_report, sources_dict)

    assert verdict.passed is False
    assert verdict.composite_score == 0.60


@pytest.mark.asyncio
async def test_rubric_judge_failing_critical_dimension(
    valid_report: Report, sample_sources: list[SourceReference]
) -> None:
    """Verify report fails if any dimension does not meet its min_pass_score, even if composite is high."""
    sources_dict = {s.source_id: s for s in sample_sources}
    mock_llm = AsyncMock()

    # High everywhere, but factual_accuracy is 0.70 (min_pass_score is 0.85)
    scores = {
        "factual_accuracy": 0.70,
        "evidence_grounding": 0.95,
        "citation_correctness": 0.95,
        "completeness": 0.95,
        "relevance": 0.95,
        "clarity_organization": 0.95,
        "logical_consistency": 0.95,
    }
    raw_eval = _create_mock_eval(scores)

    mock_llm.complete_structure.return_value = LLMManagerResponse(
        structured_response=raw_eval,
    )

    judge = RubricJudge(llm_service=mock_llm)
    verdict = await judge.evaluate(valid_report, sources_dict)

    # Composite score: 0.25*0.70 + 0.75*0.95 = 0.175 + 0.7125 = 0.8875 >= 0.85
    assert verdict.composite_score >= 0.85
    # But factual_accuracy failed
    assert verdict.dimension_scores["factual_accuracy"].passed is False
    assert verdict.passed is False


@pytest.mark.asyncio
async def test_rubric_judge_self_judging_threshold_adjustment(
    valid_report: Report, sample_sources: list[SourceReference]
) -> None:
    """Verify self-judging sets is_self_judged=True and bumps threshold from 0.85 to 0.90."""
    sources_dict = {s.source_id: s for s in sample_sources}
    mock_llm = AsyncMock()

    # Score of 0.87: passes normal 0.85 threshold, but FAILS self-judged 0.90 threshold
    scores = {
        "factual_accuracy": 0.87,
        "evidence_grounding": 0.87,
        "citation_correctness": 0.90,
        "completeness": 0.87,
        "relevance": 0.87,
        "clarity_organization": 0.87,
        "logical_consistency": 0.87,
    }
    raw_eval = _create_mock_eval(scores)

    mock_llm.complete_structure.return_value = LLMManagerResponse(
        structured_response=raw_eval,
    )

    # Both models identical -> self-judged
    judge = RubricJudge(
        llm_service=mock_llm,
        judge_model="gemini-3.5-flash-lite",
        generator_model="gemini-3.5-flash-lite",
    )

    verdict = await judge.evaluate(valid_report, sources_dict)

    assert verdict.is_self_judged is True
    assert 0.85 <= verdict.composite_score < 0.90
    assert verdict.passed is False  # Fails because 0.87 < 0.90


@pytest.mark.asyncio
async def test_rubric_judge_fallback_on_primary_failure(
    valid_report: Report, sample_sources: list[SourceReference]
) -> None:
    """Verify fallback to generator model when primary judge fails."""
    sources_dict = {s.source_id: s for s in sample_sources}
    mock_llm = AsyncMock()

    scores = {
        "factual_accuracy": 0.95,
        "evidence_grounding": 0.95,
        "citation_correctness": 0.95,
        "completeness": 0.95,
        "relevance": 0.95,
        "clarity_organization": 0.95,
        "logical_consistency": 0.95,
    }
    raw_eval = _create_mock_eval(scores)

    # First call fails, second call succeeds
    mock_llm.complete_structure.side_effect = [
        LLMManagerResponse(
            error=LLMError(
                provider="google", code="quota_exceeded", message="Judge model quota exceeded"
            )
        ),
        LLMManagerResponse(structured_response=raw_eval),
    ]

    judge = RubricJudge(
        llm_service=mock_llm,
        judge_model="gemini-2.5-pro",
        generator_model="gemini-3.5-flash-lite",
    )

    verdict = await judge.evaluate(valid_report, sources_dict)

    assert verdict.is_self_judged is True
    assert verdict.judge_model == "gemini-3.5-flash-lite"
    assert verdict.passed is True
    assert mock_llm.complete_structure.call_count == 2


@pytest.mark.asyncio
async def test_rubric_judge_unrecoverable_failure(
    valid_report: Report, sample_sources: list[SourceReference]
) -> None:
    """Verify JudgingError is raised when all model calls fail."""
    sources_dict = {s.source_id: s for s in sample_sources}
    mock_llm = AsyncMock()

    mock_llm.complete_structure.return_value = LLMManagerResponse(
        error=LLMError(provider="google", code="outage", message="Complete outage")
    )

    judge = RubricJudge(
        llm_service=mock_llm,
        judge_model="gemini-3.5-flash-lite",
        generator_model="gemini-3.5-flash-lite",
    )

    with pytest.raises(JudgingError, match="Rubric judge execution failed"):
        await judge.evaluate(valid_report, sources_dict)


@pytest.mark.asyncio
async def test_rubric_judge_missing_dimension_handled(
    valid_report: Report, sample_sources: list[SourceReference]
) -> None:
    """Verify missing dimension from judge defaults to 0.0 and fails evaluation."""
    sources_dict = {s.source_id: s for s in sample_sources}
    mock_llm = AsyncMock()

    # Only 6 dimensions provided, logical_consistency omitted
    scores = {
        "factual_accuracy": 0.95,
        "evidence_grounding": 0.95,
        "citation_correctness": 0.95,
        "completeness": 0.95,
        "relevance": 0.95,
        "clarity_organization": 0.95,
    }
    raw_eval = _create_mock_eval(scores)

    mock_llm.complete_structure.return_value = LLMManagerResponse(
        structured_response=raw_eval,
    )

    judge = RubricJudge(llm_service=mock_llm)
    verdict = await judge.evaluate(valid_report, sources_dict)

    assert "logical_consistency" in verdict.dimension_scores
    assert verdict.dimension_scores["logical_consistency"].score == 0.0
    assert verdict.dimension_scores["logical_consistency"].passed is False
    assert verdict.passed is False
