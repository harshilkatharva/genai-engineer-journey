"""Run-wide budget tracking for Plan-and-Execute Research Assistant."""

from __future__ import annotations

import time

from pydantic import BaseModel, Field


class BudgetExceededError(Exception):
    """Raised when an assistant run breaches configured token, cost, or timeout ceilings."""


class RunBudget(BaseModel):
    """Global execution budget envelope spanning all planning, execution, and replanning phases."""

    max_total_tokens: int = Field(
        default=40_000, ge=100, description="Cumulative maximum token ceiling across all LLM steps"
    )
    max_total_cost_usd: float = Field(
        default=0.25, ge=0.001, description="Cumulative maximum USD cost budget across all steps"
    )
    total_timeout_seconds: float = Field(
        default=180.0, gt=0.0, description="Maximum wall-clock execution duration in seconds"
    )
    max_concurrency: int = Field(
        default=3, ge=1, description="Maximum concurrent active tasks in DAG scheduler"
    )
    max_planning_retries: int = Field(
        default=3, ge=1, description="Maximum structural repair attempts during initial planning"
    )
    max_step_retries: int = Field(
        default=2,
        ge=0,
        description="Maximum execution retries for an individual step before replan",
    )
    max_replans: int = Field(
        default=3, ge=0, description="Maximum dynamic replanning cycles permitted per run"
    )

    current_tokens: int = Field(
        default=0, ge=0, description="Cumulative tokens consumed so far in this run"
    )
    current_cost_usd: float = Field(
        default=0.0, ge=0.0, description="Cumulative USD cost incurred so far in this run"
    )
    deadline_monotonic: float | None = Field(
        default=None, description="Monotonic timestamp marking absolute run deadline"
    )
    start_monotonic: float | None = Field(
        default=None, description="Monotonic timestamp marking run start time"
    )

    def start_timer(self) -> None:
        """Initialize monotonic start time and deadline if not already started."""
        if self.start_monotonic is None:
            now = time.monotonic()
            self.start_monotonic = now
            self.deadline_monotonic = now + self.total_timeout_seconds

    def record_usage(self, tokens: int, cost_usd: float) -> None:
        """Record token and cost consumption."""
        self.current_tokens += max(0, tokens)
        self.current_cost_usd += max(0.0, cost_usd)

    def is_token_exhausted(self) -> bool:
        """Check if cumulative tokens have reached or exceeded the ceiling."""
        return self.current_tokens >= self.max_total_tokens

    def is_cost_exhausted(self) -> bool:
        """Check if cumulative dollar cost has reached or exceeded the budget."""
        return self.current_cost_usd >= self.max_total_cost_usd

    def is_timeout_exhausted(self) -> bool:
        """Check if monotonic wall-clock deadline has passed."""
        if self.deadline_monotonic is None:
            return False
        return time.monotonic() >= self.deadline_monotonic

    def is_exhausted(self) -> bool:
        """Check if any run-wide budget limit has been reached."""
        return self.is_token_exhausted() or self.is_cost_exhausted() or self.is_timeout_exhausted()

    def check_budget(self) -> None:
        """Verify budget limits, raising BudgetExceededError if breached."""
        if self.is_token_exhausted():
            raise BudgetExceededError(
                f"Token budget exhausted: {self.current_tokens} >= {self.max_total_tokens}"
            )
        if self.is_cost_exhausted():
            raise BudgetExceededError(
                f"Cost budget exhausted: ${self.current_cost_usd:.4f} >= ${self.max_total_cost_usd:.4f}"
            )
        if self.is_timeout_exhausted():
            raise BudgetExceededError(
                f"Run deadline exceeded: wall-clock timeout of {self.total_timeout_seconds}s expired"
            )

    def remaining_seconds(self) -> float:
        """Calculate remaining wall-clock seconds before deadline."""
        if self.deadline_monotonic is None:
            return self.total_timeout_seconds
        return max(0.0, self.deadline_monotonic - time.monotonic())
