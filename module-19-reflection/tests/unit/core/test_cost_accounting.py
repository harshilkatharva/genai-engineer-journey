"""Unit tests for token-to-USD pricing and cost accounting (FINDING-03)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from genai_core import LLMManagerResponse

from self_improving_report_writer.core.budget import RunBudget
from self_improving_report_writer.core.exceptions import BudgetExceededError
from self_improving_report_writer.core.pricing import (
    MODEL_PRICING_PER_1M,
    estimate_llm_cost,
    extract_or_estimate_cost,
)
from self_improving_report_writer.generation.report_generator import ReportGenerator
from self_improving_report_writer.judging.rubric_judge import RubricJudge
from self_improving_report_writer.models.critique import (
    Critique,
    CritiqueIssue,
    IssueCategory,
    IssueSeverity,
)
from self_improving_report_writer.models.report import Report, SourceReference
from self_improving_report_writer.reflection.revision_engine import RevisionEngine


def test_estimate_llm_cost_known_models() -> None:
    """estimate_llm_cost should accurately compute USD based on model pricing."""
    # Gemini 3.5 Flash Lite: $0.075 / $0.30 per 1M
    cost_lite = estimate_llm_cost(
        "gemini-3.5-flash-lite", input_tokens=1_000_000, output_tokens=1_000_000
    )
    assert cost_lite == round(0.075 + 0.30, 6)

    # Gemini 1.5 Pro: $1.25 / $5.00 per 1M
    cost_pro = estimate_llm_cost("gemini-1.5-pro", input_tokens=10_000, output_tokens=2_000)
    expected_pro = (10_000 / 1_000_000 * 1.25) + (2_000 / 1_000_000 * 5.00)
    assert cost_pro == round(expected_pro, 6)

    # Total tokens fallback (70% input, 30% output)
    cost_total = estimate_llm_cost("gemini-1.5-flash", total_tokens=100_000)
    expected_total = (70_000 / 1_000_000 * 0.075) + (30_000 / 1_000_000 * 0.30)
    assert cost_total == round(expected_total, 6)

    # Unknown model falls back to default tier
    cost_unknown = estimate_llm_cost("custom-model-x", input_tokens=100_000, output_tokens=50_000)
    in_def, out_def = MODEL_PRICING_PER_1M["default"]
    expected_unknown = (100_000 / 1_000_000 * in_def) + (50_000 / 1_000_000 * out_def)
    assert cost_unknown == round(expected_unknown, 6)


def test_extract_or_estimate_cost_precedence() -> None:
    """Direct cost reporting should take precedence over estimation."""
    from unittest.mock import MagicMock

    # 1. Direct cost_usd on response object (e.g., custom response wrapper or mock)
    resp_direct = MagicMock()
    resp_direct.usage = {"input_tokens": 100, "output_tokens": 50}
    resp_direct.cost_usd = 0.0125
    in_t, out_t, tot_t, cost = extract_or_estimate_cost(resp_direct, model="gemini-1.5-pro")
    assert in_t == 100
    assert out_t == 50
    assert tot_t == 150
    assert cost == 0.0125

    # 2. Raw response dict with cost
    resp_raw = LLMManagerResponse(
        usage={"input_tokens": 200, "output_tokens": 100},
        raw_response={"cost": 0.035},
    )
    _, _, _, cost_raw = extract_or_estimate_cost(resp_raw, model="gemini-1.5-pro")
    assert cost_raw == 0.035

    # 3. No cost reported -> fallback estimation
    resp_none = LLMManagerResponse(
        usage={"input_tokens": 100_000, "output_tokens": 20_000},
        raw_response=None,
    )
    _, _, _, cost_est = extract_or_estimate_cost(resp_none, model="gemini-1.5-pro")
    assert cost_est > 0.0
    assert cost_est == estimate_llm_cost(
        "gemini-1.5-pro", input_tokens=100_000, output_tokens=20_000
    )


def test_run_budget_exhaustion_on_cost_limit_with_estimated_cost() -> None:
    """RunBudget should record estimated cost and halt when max_cost_usd is exceeded."""
    budget = RunBudget(max_tokens=1_000_000, max_cost_usd=0.01)
    assert not budget.is_exhausted()

    # Record usage with model: 10,000 pro tokens costs > $0.01
    budget.record_usage(
        tokens=10_000,
        model="gemini-1.5-pro",
        input_tokens=8_000,
        output_tokens=2_000,
    )

    assert budget.used_cost_usd > 0.01
    assert budget.is_exhausted()
    breach = budget.check_budget_exceeded()
    assert breach is not None
    assert "COST_EXHAUSTED" in breach

    with pytest.raises(BudgetExceededError, match="COST_EXHAUSTED"):
        budget.assert_within_budget()


@pytest.mark.asyncio
async def test_report_generator_computes_cost_when_response_lacks_cost(
    valid_report: Report, sample_sources: list[SourceReference]
) -> None:
    """ReportGenerator should compute non-zero cost when response has usage but no cost."""
    mock_llm = AsyncMock()
    mock_llm.complete_structure.return_value = LLMManagerResponse(
        structured_response=valid_report,
        usage={"input_tokens": 5_000, "output_tokens": 1_500, "total_tokens": 6_500},
        raw_response="SDK Response Object (not a dict)",
    )

    generator = ReportGenerator(
        llm_service=mock_llm,
        model="gemini-1.5-flash",
    )

    _, tokens, cost = await generator.generate_report("topic", sample_sources)
    assert tokens == 6_500
    assert cost > 0.0
    assert cost == estimate_llm_cost("gemini-1.5-flash", input_tokens=5_000, output_tokens=1_500)


@pytest.mark.asyncio
async def test_rubric_judge_computes_cost_when_response_lacks_cost(
    valid_report: Report, sample_sources: list[SourceReference]
) -> None:
    """RubricJudge should compute non-zero cost when judge response has usage but no cost."""
    dim_names = [
        "factual_accuracy",
        "evidence_grounding",
        "citation_correctness",
        "completeness",
        "relevance",
        "clarity_organization",
        "logical_consistency",
    ]
    raw_eval = {
        "dimension_scores": [
            {"dimension_name": d, "score": 0.90, "passed": True, "reasoning": "Good"}
            for d in dim_names
        ],
        "overall_feedback": "Great report",
    }

    mock_llm = AsyncMock()
    mock_llm.complete_structure.return_value = LLMManagerResponse(
        structured_response=raw_eval,
        usage={"input_tokens": 3_000, "output_tokens": 800, "total_tokens": 3_800},
        raw_response="SDK Response Object (not a dict)",
    )

    judge = RubricJudge(
        llm_service=mock_llm,
        judge_model="gemini-1.5-pro",
    )

    sources_dict = {s.source_id: s for s in sample_sources}
    verdict = await judge.evaluate(valid_report, sources_dict)

    assert verdict.tokens_used == 3_800
    assert verdict.cost_usd > 0.0
    assert verdict.cost_usd == estimate_llm_cost(
        "gemini-1.5-pro", input_tokens=3_000, output_tokens=800
    )


@pytest.mark.asyncio
async def test_revision_engine_computes_cost_when_response_lacks_cost(
    valid_report: Report, sample_sources: list[SourceReference]
) -> None:
    """RevisionEngine should compute non-zero cost when revision response has usage but no cost."""
    mock_llm = AsyncMock()
    mock_llm.complete_structure.return_value = LLMManagerResponse(
        structured_response=valid_report,
        usage={"input_tokens": 4_000, "output_tokens": 1_200, "total_tokens": 5_200},
        raw_response=None,
    )

    engine = RevisionEngine(
        llm_service=mock_llm,
        model="gemini-1.5-flash",
    )

    critique = Critique(
        round_number=1,
        summary="Needs revision",
        should_revise=True,
        issues=[
            CritiqueIssue(
                issue_id="issue_1",
                severity=IssueSeverity.MAJOR,
                category=IssueCategory.FACTUAL_GROUNDING,
                explanation="Need more data",
                suggested_correction="Add statistics",
            )
        ],
    )

    _, tokens, cost = await engine.revise_report(valid_report, sample_sources, critique)
    assert tokens == 5_200
    assert cost > 0.0
    assert cost == estimate_llm_cost("gemini-1.5-flash", input_tokens=4_000, output_tokens=1_200)
