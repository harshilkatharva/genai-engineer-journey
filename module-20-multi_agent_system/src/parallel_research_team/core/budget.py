"""Concurrency-safe shared execution budget for Parallel Research Team."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from ..exceptions import BudgetExceededError
from .config import ResearchTeamConfig, get_research_team_config


class SharedRunBudget:
    """Thread- and async-safe global budget envelope shared across parallel workers and lead agent.

    Protects concurrent token and cost accounting using an asyncio.Lock to prevent race
    conditions when multiple parallel workers record usage simultaneously.
    """

    def __init__(
        self,
        config: ResearchTeamConfig | None = None,
        max_total_tokens: int | None = None,
        max_total_cost_usd: float | None = None,
        max_wall_clock_seconds: float | None = None,
    ) -> None:
        cfg = config or get_research_team_config()
        self.max_total_tokens = (
            max_total_tokens if max_total_tokens is not None else cfg.max_total_tokens
        )
        self.max_total_cost_usd = (
            max_total_cost_usd if max_total_cost_usd is not None else cfg.max_total_cost_usd
        )
        self.max_wall_clock_seconds = (
            max_wall_clock_seconds
            if max_wall_clock_seconds is not None
            else cfg.max_wall_clock_seconds
        )

        self.current_tokens: int = 0
        self.current_cost_usd: float = 0.0
        self.start_monotonic: float | None = None
        self.deadline_monotonic: float | None = None
        self._lock = asyncio.Lock()

    def start_timer(self) -> None:
        """Start wall-clock execution timer and set monotonic deadline."""
        if self.start_monotonic is None:
            now = time.monotonic()
            self.start_monotonic = now
            self.deadline_monotonic = now + self.max_wall_clock_seconds

    @property
    def elapsed_seconds(self) -> float:
        """Seconds elapsed since timer start."""
        if self.start_monotonic is None:
            return 0.0
        return round(time.monotonic() - self.start_monotonic, 3)

    def is_deadline_exceeded(self) -> bool:
        """Check if monotonic deadline has passed."""
        if self.deadline_monotonic is not None:
            return time.monotonic() >= self.deadline_monotonic
        return False

    def is_token_budget_exhausted(self) -> bool:
        """Check if token ceiling has been breached."""
        return self.current_tokens >= self.max_total_tokens

    def is_cost_budget_exhausted(self) -> bool:
        """Check if cost ceiling has been breached."""
        return self.current_cost_usd >= self.max_total_cost_usd

    async def can_proceed(self) -> bool:
        """Concurrency-safe check whether another operation can be launched."""
        async with self._lock:
            if self.is_deadline_exceeded():
                return False
            if self.is_token_budget_exhausted():
                return False
            return not self.is_cost_budget_exhausted()

    async def record_usage(self, tokens: int = 0, cost_usd: float = 0.0) -> None:
        """Atomically record resource consumption from a worker or lead agent call."""
        async with self._lock:
            self.current_tokens += max(0, tokens)
            self.current_cost_usd = round(self.current_cost_usd + max(0.0, cost_usd), 6)

    async def check_budget(self) -> None:
        """Raise BudgetExceededError if any resource ceiling has been exceeded."""
        async with self._lock:
            if self.is_deadline_exceeded():
                raise BudgetExceededError(
                    f"Execution deadline exceeded: elapsed {self.elapsed_seconds:.1f}s >= "
                    f"limit {self.max_wall_clock_seconds:.1f}s"
                )
            if self.is_token_budget_exhausted():
                raise BudgetExceededError(
                    f"Token budget exhausted: used {self.current_tokens} >= "
                    f"limit {self.max_total_tokens}"
                )
            if self.is_cost_budget_exhausted():
                raise BudgetExceededError(
                    f"Cost budget exhausted: spent ${self.current_cost_usd:.4f} >= "
                    f"limit ${self.max_total_cost_usd:.4f}"
                )

    async def get_remaining_budget(self) -> dict[str, Any]:
        """Return remaining resources."""
        async with self._lock:
            rem_tokens = max(0, self.max_total_tokens - self.current_tokens)
            rem_cost = max(0.0, round(self.max_total_cost_usd - self.current_cost_usd, 6))
            rem_seconds = 0.0
            if self.deadline_monotonic is not None:
                rem_seconds = max(0.0, round(self.deadline_monotonic - time.monotonic(), 3))
            return {
                "remaining_tokens": rem_tokens,
                "remaining_cost_usd": rem_cost,
                "remaining_seconds": rem_seconds,
            }

    async def get_summary(self) -> dict[str, Any]:
        """Return a snapshot of current budget consumption."""
        async with self._lock:
            return {
                "current_tokens": self.current_tokens,
                "max_total_tokens": self.max_total_tokens,
                "current_cost_usd": self.current_cost_usd,
                "max_total_cost_usd": self.max_total_cost_usd,
                "elapsed_seconds": self.elapsed_seconds,
                "max_wall_clock_seconds": self.max_wall_clock_seconds,
                "is_deadline_exceeded": self.is_deadline_exceeded(),
                "is_token_exhausted": self.is_token_budget_exhausted(),
                "is_cost_exhausted": self.is_cost_budget_exhausted(),
            }
