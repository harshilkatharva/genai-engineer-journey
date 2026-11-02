"""Unit tests for RunBudget and BudgetSummary multi-resource tracking."""

from __future__ import annotations

import pytest

from self_improving_report_writer.core.budget import BudgetSummary, RunBudget
from self_improving_report_writer.core.config import ReportWriterConfig
from self_improving_report_writer.core.exceptions import BudgetExceededError


def test_budget_initialization_defaults() -> None:
    """RunBudget should initialize with default config parameters."""
    cfg = ReportWriterConfig(
        max_total_tokens=10000,
        max_total_cost_usd=0.15,
        max_wall_clock_seconds=30.0,
        max_reflection_rounds=2,
    )
    budget = RunBudget(config=cfg)

    assert budget.max_tokens == 10000
    assert budget.max_cost_usd == 0.15
    assert budget.max_seconds == 30.0
    assert budget.max_rounds == 2
    assert budget.used_tokens == 0
    assert budget.used_cost_usd == 0.0
    assert budget.rounds_executed == 0
    assert not budget.is_exhausted()


def test_budget_usage_recording_and_remaining() -> None:
    """RunBudget should accurately deduct token and financial expenditures."""
    budget = RunBudget(max_tokens=5000, max_cost_usd=0.10, max_seconds=60.0, max_rounds=3)

    budget.record_usage(tokens=1500, cost_usd=0.03)
    assert budget.used_tokens == 1500
    assert budget.remaining_tokens == 3500
    assert budget.used_cost_usd == 0.03
    assert budget.remaining_cost_usd == 0.07
    assert not budget.is_exhausted()

    budget.record_round_completed()
    assert budget.rounds_executed == 1


def test_budget_token_exhaustion() -> None:
    """RunBudget should detect token ceiling breach and raise on assert."""
    budget = RunBudget(max_tokens=2000, max_cost_usd=1.0)
    budget.record_usage(tokens=2500, cost_usd=0.01)

    assert budget.is_exhausted()
    breach = budget.check_budget_exceeded()
    assert breach is not None
    assert "TOKENS_EXHAUSTED" in breach

    with pytest.raises(BudgetExceededError, match="TOKENS_EXHAUSTED"):
        budget.assert_within_budget()


def test_budget_cost_exhaustion() -> None:
    """RunBudget should detect cost ceiling breach and raise on assert."""
    budget = RunBudget(max_tokens=50000, max_cost_usd=0.05)
    budget.record_usage(tokens=100, cost_usd=0.06)

    assert budget.is_exhausted()
    breach = budget.check_budget_exceeded()
    assert breach is not None
    assert "COST_EXHAUSTED" in breach

    with pytest.raises(BudgetExceededError, match="COST_EXHAUSTED"):
        budget.assert_within_budget()


def test_budget_round_limit_exhaustion() -> None:
    """RunBudget should detect max rounds reached."""
    budget = RunBudget(max_rounds=2)
    budget.record_round_completed()
    assert not budget.is_exhausted()

    budget.record_round_completed()
    assert budget.is_exhausted()
    breach = budget.check_budget_exceeded()
    assert breach is not None
    assert "MAX_ROUNDS_REACHED" in breach


def test_budget_to_summary() -> None:
    """RunBudget.to_summary should produce a valid immutable BudgetSummary."""
    budget = RunBudget(max_tokens=1000, max_cost_usd=0.05, max_seconds=10.0, max_rounds=2)
    budget.record_usage(tokens=400, cost_usd=0.01)
    budget.record_round_completed()

    summary = budget.to_summary()
    assert isinstance(summary, BudgetSummary)
    assert summary.used_tokens == 400
    assert summary.max_tokens == 1000
    assert summary.used_cost_usd == 0.01
    assert summary.max_cost_usd == 0.05
    assert summary.rounds_executed == 1
    assert summary.max_rounds == 2
    assert not summary.is_exhausted
    assert summary.exhaustion_reason is None
