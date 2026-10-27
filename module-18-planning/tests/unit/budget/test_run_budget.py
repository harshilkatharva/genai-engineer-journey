"""Unit tests for RunBudget lifecycle and ceiling enforcement."""

from __future__ import annotations

import time

import pytest

from plan_and_execute_research_assistant.models import (
    BudgetExceededError,
    RunBudget,
)


def test_budget_defaults_and_start_timer() -> None:
    """Verify default limits and timer initialization."""
    budget = RunBudget()
    assert budget.max_total_tokens == 40_000
    assert budget.max_total_cost_usd == pytest.approx(0.25)
    assert budget.total_timeout_seconds == pytest.approx(180.0)
    assert budget.start_monotonic is None
    assert budget.deadline_monotonic is None

    budget.start_timer()
    assert budget.start_monotonic is not None
    assert budget.deadline_monotonic is not None
    assert budget.deadline_monotonic > budget.start_monotonic


def test_budget_record_usage_accumulates() -> None:
    """Verify usage records accumulate tokens and cost."""
    budget = RunBudget()
    budget.start_timer()

    budget.record_usage(tokens=1000, cost_usd=0.015)
    budget.record_usage(tokens=500, cost_usd=0.010)

    assert budget.current_tokens == 1500
    assert budget.current_cost_usd == pytest.approx(0.025)
    assert not budget.is_exhausted()


def test_budget_token_exhaustion_raises() -> None:
    """Verify check_budget raises BudgetExceededError when token ceiling is reached."""
    budget = RunBudget(max_total_tokens=2000)
    budget.start_timer()
    budget.record_usage(tokens=2001, cost_usd=0.01)

    assert budget.is_token_exhausted()
    assert budget.is_exhausted()

    with pytest.raises(BudgetExceededError) as exc_info:
        budget.check_budget()
    assert "Token budget exhausted" in str(exc_info.value)


def test_budget_cost_exhaustion_raises() -> None:
    """Verify check_budget raises BudgetExceededError when cost ceiling is reached."""
    budget = RunBudget(max_total_cost_usd=0.05)
    budget.start_timer()
    budget.record_usage(tokens=500, cost_usd=0.055)

    assert budget.is_cost_exhausted()
    assert budget.is_exhausted()

    with pytest.raises(BudgetExceededError) as exc_info:
        budget.check_budget()
    assert "Cost budget exhausted" in str(exc_info.value)


def test_budget_timeout_exhaustion_raises() -> None:
    """Verify check_budget raises BudgetExceededError when wall-clock deadline expires."""
    budget = RunBudget(total_timeout_seconds=0.05)
    budget.start_timer()

    # Sleep slightly past deadline
    time.sleep(0.06)

    assert budget.is_timeout_exhausted()
    assert budget.is_exhausted()

    with pytest.raises(BudgetExceededError) as exc_info:
        budget.check_budget()
    assert "Run deadline exceeded" in str(exc_info.value)


def test_budget_remaining_seconds() -> None:
    """Verify remaining_seconds accurately reports time remaining."""
    budget = RunBudget(total_timeout_seconds=10.0)
    assert budget.remaining_seconds() == pytest.approx(10.0)

    budget.start_timer()
    rem = budget.remaining_seconds()
    assert 0.0 < rem <= 10.0
