"""RunBudget: multi-resource budget enforcement across reflection rounds."""

from __future__ import annotations

import logging
import time

from genai_core import ObservabilityEvent, safe_log
from pydantic import BaseModel, Field

from self_improving_report_writer.core.config import ReportWriterConfig
from self_improving_report_writer.core.exceptions import BudgetExceededError

logger = logging.getLogger(__name__)


class BudgetSummary(BaseModel):
    """Immutable snapshot of budget consumption at completion or termination."""

    used_tokens: int = Field(ge=0, description="Total tokens consumed across all phases")
    max_tokens: int = Field(ge=1, description="Configured token ceiling")
    used_cost_usd: float = Field(ge=0.0, description="Total USD expenditure")
    max_cost_usd: float = Field(ge=0.0, description="Configured financial ceiling in USD")
    elapsed_seconds: float = Field(ge=0.0, description="Wall-clock elapsed execution seconds")
    max_seconds: float = Field(ge=0.0, description="Configured wall-clock deadline in seconds")
    rounds_executed: int = Field(ge=0, description="Total reflection rounds executed")
    max_rounds: int = Field(ge=1, description="Configured maximum reflection rounds")
    is_exhausted: bool = Field(description="Whether any resource limit was reached or exceeded")
    exhaustion_reason: str | None = Field(
        default=None, description="Resource identifying breach: TOKENS, COST, TIME, or ROUNDS"
    )


class RunBudget:
    """Multi-resource bounded envelope tracking tokens, cost, execution time, and reflection rounds."""

    def __init__(
        self,
        config: ReportWriterConfig | None = None,
        max_tokens: int | None = None,
        max_cost_usd: float | None = None,
        max_seconds: float | None = None,
        max_rounds: int | None = None,
    ) -> None:
        cfg = config or ReportWriterConfig()
        self.max_tokens = max_tokens if max_tokens is not None else cfg.max_total_tokens
        self.max_cost_usd = max_cost_usd if max_cost_usd is not None else cfg.max_total_cost_usd
        self.max_seconds = max_seconds if max_seconds is not None else cfg.max_wall_clock_seconds
        self.max_rounds = max_rounds if max_rounds is not None else cfg.max_reflection_rounds

        self.used_tokens = 0
        self.used_cost_usd = 0.0
        self.start_time = time.monotonic()
        self.rounds_executed = 0

    @property
    def elapsed_seconds(self) -> float:
        """Wall-clock seconds elapsed since budget creation."""
        return round(time.monotonic() - self.start_time, 3)

    @property
    def remaining_tokens(self) -> int:
        """Unconsumed token allowance remaining in budget."""
        return max(0, self.max_tokens - self.used_tokens)

    @property
    def remaining_cost_usd(self) -> float:
        """Unconsumed financial budget remaining in USD."""
        return max(0.0, round(self.max_cost_usd - self.used_cost_usd, 6))

    @property
    def remaining_seconds(self) -> float:
        """Execution time remaining before wall-clock deadline."""
        return max(0.0, round(self.max_seconds - self.elapsed_seconds, 3))

    def record_usage(self, tokens: int = 0, cost_usd: float = 0.0) -> None:
        """Record token consumption and dollar expenditure."""
        self.used_tokens += max(0, tokens)
        self.used_cost_usd = round(self.used_cost_usd + max(0.0, cost_usd), 6)

        if self.is_exhausted():
            reason = self.check_budget_exceeded()
            safe_log(
                logger,
                logging.WARNING,
                f"RunBudget resource limit reached: {reason} "
                f"(tokens={self.used_tokens}/{self.max_tokens}, "
                f"cost=${self.used_cost_usd:.4f}/${self.max_cost_usd:.2f}, "
                f"time={self.elapsed_seconds:.1f}s/{self.max_seconds:.1f}s)",
                event=ObservabilityEvent.LLM_STREAM_FAILED,
            )

    def record_round_completed(self) -> None:
        """Increment count of completed reflection rounds."""
        self.rounds_executed += 1

    def is_exhausted(self, current_round: int | None = None) -> bool:
        """Check whether any multi-resource budget ceiling is exhausted."""
        return self.check_budget_exceeded(current_round=current_round) is not None

    def check_budget_exceeded(self, current_round: int | None = None) -> str | None:
        """Identify which resource threshold was breached, or None if within bounds."""
        if self.used_tokens >= self.max_tokens:
            return f"TOKENS_EXHAUSTED (used {self.used_tokens} >= max {self.max_tokens})"
        if self.used_cost_usd >= self.max_cost_usd:
            return (
                f"COST_EXHAUSTED (used ${self.used_cost_usd:.4f} >= max ${self.max_cost_usd:.2f})"
            )
        if self.elapsed_seconds >= self.max_seconds:
            return f"TIME_EXHAUSTED (elapsed {self.elapsed_seconds:.1f}s >= max {self.max_seconds:.1f}s)"
        round_val = current_round if current_round is not None else self.rounds_executed
        if round_val >= self.max_rounds:
            return f"MAX_ROUNDS_REACHED (executed {round_val} >= max {self.max_rounds})"
        return None

    def assert_within_budget(self, current_round: int | None = None) -> None:
        """Raise BudgetExceededError if any resource ceiling has been exceeded."""
        breach = self.check_budget_exceeded(current_round=current_round)
        if breach:
            raise BudgetExceededError(f"Reflection run aborted due to budget exhaustion: {breach}")

    def to_summary(self) -> BudgetSummary:
        """Generate an immutable summary of budget metrics."""
        exhaustion_reason = self.check_budget_exceeded()
        return BudgetSummary(
            used_tokens=self.used_tokens,
            max_tokens=self.max_tokens,
            used_cost_usd=round(self.used_cost_usd, 6),
            max_cost_usd=self.max_cost_usd,
            elapsed_seconds=self.elapsed_seconds,
            max_seconds=self.max_seconds,
            rounds_executed=self.rounds_executed,
            max_rounds=self.max_rounds,
            is_exhausted=exhaustion_reason is not None,
            exhaustion_reason=exhaustion_reason,
        )
