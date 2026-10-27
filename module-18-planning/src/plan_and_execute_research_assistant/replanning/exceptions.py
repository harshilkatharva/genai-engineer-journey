"""Exceptions for dynamic replanning and failure recovery."""

from __future__ import annotations


class ReplanningError(Exception):
    """Base exception for replanning failures."""


class ReplanningLimitExceededError(ReplanningError):
    """Raised when dynamic replanning attempts exceed the configured ceiling."""

    def __init__(self, current_replans: int, max_replans: int, goal: str) -> None:
        self.current_replans = current_replans
        self.max_replans = max_replans
        self.goal = goal
        super().__init__(
            f"Replanning ceiling exceeded ({current_replans}/{max_replans}) for goal: '{goal}'. "
            "Execution halted to prevent infinite loops and runaway token expenditure."
        )
