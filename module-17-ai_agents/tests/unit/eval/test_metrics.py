"""Unit tests for benchmark metrics evaluators."""

from __future__ import annotations

import pytest
from genai_core.models.evaluation_model import EvalOutput, EvalSample, ExpectedAnswer

from research_and_support_agent.eval.metrics import (
    CostBudgetMetric,
    KeywordComplianceMetric,
    StepBudgetMetric,
)


@pytest.mark.asyncio
async def test_step_budget_metric_pass_and_fail() -> None:
    """Verify StepBudgetMetric evaluates step counts against allowed limit."""
    metric = StepBudgetMetric(max_allowed_steps=3)
    sample = EvalSample(id="sample_step", query="test")

    # Pass: 2 steps <= 3
    pass_out = EvalOutput(token_usage={"steps": 2})
    res_pass = await metric.evaluate(sample, pass_out)
    assert res_pass.passed is True
    assert res_pass.score == 1.0

    # Fail: 5 steps > 3
    fail_out = EvalOutput(token_usage={"steps": 5})
    res_fail = await metric.evaluate(sample, fail_out)
    assert res_fail.passed is False
    assert res_fail.score == 0.0


@pytest.mark.asyncio
async def test_cost_budget_metric_pass_and_fail() -> None:
    """Verify CostBudgetMetric evaluates cost against allowed monetary budget."""
    metric = CostBudgetMetric(max_allowed_cost=0.01)
    sample = EvalSample(id="sample_cost", query="test")

    # Pass: $0.005 <= $0.01
    pass_out = EvalOutput(raw_response={"cost_usd": 0.005})
    res_pass = await metric.evaluate(sample, pass_out)
    assert res_pass.passed is True

    # Fail: $0.02 > $0.01
    fail_out = EvalOutput(raw_response={"cost_usd": 0.02})
    res_fail = await metric.evaluate(sample, fail_out)
    assert res_fail.passed is False


@pytest.mark.asyncio
async def test_keyword_compliance_metric() -> None:
    """Verify KeywordComplianceMetric evaluates mandatory required keywords."""
    metric = KeywordComplianceMetric()
    sample = EvalSample(
        id="sample_kw",
        query="Check warranty",
        expected_answer=ExpectedAnswer(required=["30", "warranty"]),
    )

    # Full match: both keywords present
    full_match = EvalOutput(answer="The policy includes a 30-day warranty window.")
    res_full = await metric.evaluate(sample, full_match)
    assert res_full.passed is True
    assert res_full.score == 1.0

    # Partial match: only 1 keyword present
    partial_match = EvalOutput(answer="The policy specifies a 30-day return window.")
    res_partial = await metric.evaluate(sample, partial_match)
    assert res_partial.passed is False
    assert res_partial.score == 0.5
