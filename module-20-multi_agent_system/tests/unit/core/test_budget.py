"""Unit tests for concurrency-safe SharedRunBudget."""

from __future__ import annotations

import asyncio

import pytest

from parallel_research_team.core.budget import SharedRunBudget
from parallel_research_team.exceptions import BudgetExceededError


@pytest.mark.asyncio
async def test_budget_initialization() -> None:
    budget = SharedRunBudget(
        max_total_tokens=1000,
        max_total_cost_usd=0.05,
        max_wall_clock_seconds=10.0,
    )
    budget.start_timer()

    assert budget.current_tokens == 0
    assert budget.current_cost_usd == 0.0
    assert await budget.can_proceed() is True
    assert budget.elapsed_seconds >= 0.0

    rem = await budget.get_remaining_budget()
    assert rem["remaining_tokens"] == 1000
    assert rem["remaining_cost_usd"] == 0.05


@pytest.mark.asyncio
async def test_concurrent_usage_recording() -> None:
    """Stress test: 50 concurrent worker tasks recording usage simultaneously."""
    budget = SharedRunBudget(
        max_total_tokens=100_000,
        max_total_cost_usd=1.0,
        max_wall_clock_seconds=60.0,
    )
    budget.start_timer()

    num_tasks = 50
    tokens_per_task = 100
    cost_per_task = 0.001

    async def worker_record() -> None:
        await budget.record_usage(tokens=tokens_per_task, cost_usd=cost_per_task)

    tasks = [worker_record() for _ in range(num_tasks)]
    await asyncio.gather(*tasks)

    summary = await budget.get_summary()
    assert summary["current_tokens"] == num_tasks * tokens_per_task
    assert summary["current_cost_usd"] == pytest.approx(num_tasks * cost_per_task, rel=1e-5)


@pytest.mark.asyncio
async def test_token_exhaustion_detection() -> None:
    budget = SharedRunBudget(
        max_total_tokens=500,
        max_total_cost_usd=1.0,
        max_wall_clock_seconds=60.0,
    )
    budget.start_timer()

    await budget.record_usage(tokens=400)
    assert await budget.can_proceed() is True

    await budget.record_usage(tokens=100)
    assert await budget.can_proceed() is False
    assert budget.is_token_budget_exhausted() is True

    with pytest.raises(BudgetExceededError, match="Token budget exhausted"):
        await budget.check_budget()


@pytest.mark.asyncio
async def test_cost_exhaustion_detection() -> None:
    budget = SharedRunBudget(
        max_total_tokens=10_000,
        max_total_cost_usd=0.02,
        max_wall_clock_seconds=60.0,
    )
    budget.start_timer()

    await budget.record_usage(cost_usd=0.015)
    assert await budget.can_proceed() is True

    await budget.record_usage(cost_usd=0.01)
    assert await budget.can_proceed() is False
    assert budget.is_cost_budget_exhausted() is True

    with pytest.raises(BudgetExceededError, match="Cost budget exhausted"):
        await budget.check_budget()


@pytest.mark.asyncio
async def test_deadline_exceeded() -> None:
    budget = SharedRunBudget(
        max_total_tokens=10_000,
        max_total_cost_usd=1.0,
        max_wall_clock_seconds=0.05,
    )
    budget.start_timer()

    # Sleep slightly past deadline
    await asyncio.sleep(0.06)

    assert budget.is_deadline_exceeded() is True
    assert await budget.can_proceed() is False

    with pytest.raises(BudgetExceededError, match="Execution deadline exceeded"):
        await budget.check_budget()


def test_budget_resumption_initial_elapsed_seconds() -> None:
    budget = SharedRunBudget(
        max_total_tokens=10_000,
        max_total_cost_usd=1.0,
        max_wall_clock_seconds=180.0,
        initial_elapsed_seconds=150.0,
    )
    assert budget.elapsed_seconds == 150.0

    budget.start_timer()
    assert budget.start_monotonic is not None
    assert budget.deadline_monotonic is not None
    # Remaining time should be ~30.0s, not 180.0s
    remaining = budget.deadline_monotonic - budget.start_monotonic
    assert abs(remaining - 30.0) < 0.1
    assert budget.elapsed_seconds >= 150.0
