"""Unit tests for model diversity enforcement across generation, judging, and revision (FINDING-02)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from genai_core import LLMError, LLMManagerResponse

from self_improving_report_writer.generation.report_generator import ReportGenerator
from self_improving_report_writer.judging.prompts import LLMJudgeEvaluation
from self_improving_report_writer.judging.rubric_judge import RubricJudge
from self_improving_report_writer.models.critique import (
    Critique,
    CritiqueIssue,
    IssueCategory,
    IssueSeverity,
)
from self_improving_report_writer.models.report import Report, SourceReference
from self_improving_report_writer.models.rubric import DimensionScore
from self_improving_report_writer.reflection.revision_engine import RevisionEngine


def _create_mock_eval(score: float = 0.90) -> LLMJudgeEvaluation:
    dim_names = [
        "factual_accuracy",
        "evidence_grounding",
        "citation_correctness",
        "completeness",
        "relevance",
        "clarity_organization",
        "logical_consistency",
    ]
    return LLMJudgeEvaluation(
        dimension_scores=[
            DimensionScore(
                dimension_name=d,
                score=score,
                passed=score >= 0.80,
                reasoning="Solid evidence and reasoning",
            )
            for d in dim_names
        ],
        overall_feedback="Well supported and aligned.",
    )


@pytest.mark.asyncio
async def test_rubric_judge_passes_judge_model_and_provider_to_llm_service(
    valid_report: Report, sample_sources: list[SourceReference]
) -> None:
    """RubricJudge must forward target_model and target_provider to LLMService."""
    sources_dict = {s.source_id: s for s in sample_sources}
    mock_llm = AsyncMock()
    mock_llm.complete_structure.return_value = LLMManagerResponse(
        structured_response=_create_mock_eval(0.92),
        model="gemini-1.5-pro",
    )

    judge = RubricJudge(
        llm_service=mock_llm,
        judge_model="gemini-1.5-pro",
        generator_model="gemini-1.5-flash",
        judge_provider="google",
    )

    verdict = await judge.evaluate(valid_report, sources_dict)

    assert verdict.passed is True
    assert verdict.is_self_judged is False
    assert verdict.judge_model == "gemini-1.5-pro"

    # Verify LLMService was called with explicit model and provider
    call_kwargs = mock_llm.complete_structure.call_args.kwargs
    assert call_kwargs.get("model") == "gemini-1.5-pro"
    assert call_kwargs.get("provider") == "google"


@pytest.mark.asyncio
async def test_rubric_judge_fallback_passes_generator_model_and_engages_self_judging(
    valid_report: Report, sample_sources: list[SourceReference]
) -> None:
    """When primary judge fails, fallback must use generator_model and flag is_self_judged=True."""
    sources_dict = {s.source_id: s for s in sample_sources}
    mock_llm = AsyncMock()
    mock_llm.complete_structure.side_effect = [
        LLMManagerResponse(
            error=LLMError(provider="google", code="rate_limit", message="Quota exceeded")
        ),
        LLMManagerResponse(
            structured_response=_create_mock_eval(0.93),
            model="gemini-1.5-flash",
        ),
    ]

    judge = RubricJudge(
        llm_service=mock_llm,
        judge_model="gemini-1.5-pro",
        generator_model="gemini-1.5-flash",
    )

    verdict = await judge.evaluate(valid_report, sources_dict)

    assert verdict.is_self_judged is True
    assert verdict.judge_model == "gemini-1.5-flash"
    assert verdict.passed is True

    # First call targeted judge_model
    first_call_kwargs = mock_llm.complete_structure.call_args_list[0].kwargs
    assert first_call_kwargs.get("model") == "gemini-1.5-pro"

    # Fallback call targeted generator_model
    second_call_kwargs = mock_llm.complete_structure.call_args_list[1].kwargs
    assert second_call_kwargs.get("model") == "gemini-1.5-flash"


@pytest.mark.asyncio
async def test_rubric_judge_dynamically_detects_self_judging_when_returned_model_matches_generator(
    valid_report: Report, sample_sources: list[SourceReference]
) -> None:
    """If provider returns the generator model despite requesting judge model, detect self-judging and raise threshold."""
    sources_dict = {s.source_id: s for s in sample_sources}
    mock_llm = AsyncMock()

    # Score of 0.87 passes base threshold (0.85) but fails self-judged threshold (0.90)
    mock_llm.complete_structure.return_value = LLMManagerResponse(
        structured_response=_create_mock_eval(0.87),
        model="gemini-1.5-flash",  # Matches generator_model!
    )

    judge = RubricJudge(
        llm_service=mock_llm,
        judge_model="gemini-1.5-pro",
        generator_model="gemini-1.5-flash",
    )

    verdict = await judge.evaluate(valid_report, sources_dict)

    assert verdict.is_self_judged is True
    assert verdict.judge_model == "gemini-1.5-flash"
    assert 0.85 <= verdict.composite_score < 0.90
    assert verdict.passed is False  # Rejected due to +0.05 self-judging penalty bump!


@pytest.mark.asyncio
async def test_report_generator_passes_generator_model_and_provider(
    valid_report: Report, sample_sources: list[SourceReference]
) -> None:
    """ReportGenerator must pass model and provider to LLMService.complete_structure."""
    mock_llm = AsyncMock()
    mock_llm.complete_structure.return_value = LLMManagerResponse(
        structured_response=valid_report,
        model="gemini-1.5-flash",
    )

    generator = ReportGenerator(
        llm_service=mock_llm,
        provider="google",
        model="gemini-1.5-flash",
    )

    report, _tokens, _cost = await generator.generate_report(
        topic="Quantum Computing",
        sources=sample_sources,
    )

    assert report.topic == valid_report.topic
    call_kwargs = mock_llm.complete_structure.call_args.kwargs
    assert call_kwargs.get("model") == "gemini-1.5-flash"
    assert call_kwargs.get("provider") == "google"


@pytest.mark.asyncio
async def test_revision_engine_passes_revision_model_and_provider(
    valid_report: Report, sample_sources: list[SourceReference]
) -> None:
    """RevisionEngine must pass model and provider to LLMService.complete_structure."""
    mock_llm = AsyncMock()
    mock_llm.complete_structure.return_value = LLMManagerResponse(
        structured_response=valid_report,
        model="gemini-1.5-flash",
    )

    engine = RevisionEngine(
        llm_service=mock_llm,
        provider="google",
        model="gemini-1.5-flash",
    )

    critique = Critique(
        round_number=1,
        summary="Needs revision",
        should_revise=True,
        issues=[
            CritiqueIssue(
                issue_id="issue_1",
                severity=IssueSeverity.CRITICAL,
                category=IssueCategory.FACTUAL_GROUNDING,
                explanation="Claim has no citation in Section 1",
                suggested_correction="Add citation",
            )
        ],
    )

    report, _tokens, _cost = await engine.revise_report(
        current_report=valid_report,
        sources=sample_sources,
        critique=critique,
    )

    assert report.topic == valid_report.topic
    call_kwargs = mock_llm.complete_structure.call_args.kwargs
    assert call_kwargs.get("model") == "gemini-1.5-flash"
    assert call_kwargs.get("provider") == "google"
