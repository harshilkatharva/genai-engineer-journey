"""Exceptions related to human plan review and approval."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plan_and_execute_research_assistant.planning.exceptions import PlanValidationError


class PlanRejectedError(Exception):
    """Raised when an operator explicitly rejects a proposed plan."""

    def __init__(self, message: str, reviewer_id: str, reason: str | None = None) -> None:
        super().__init__(message)
        self.reviewer_id = reviewer_id
        self.reason = reason


class ReviewValidationError(Exception):
    """Raised when a reviewer-modified plan fails structural DAG validation."""

    def __init__(self, message: str, validation_error: PlanValidationError) -> None:
        super().__init__(message)
        self.validation_error = validation_error
