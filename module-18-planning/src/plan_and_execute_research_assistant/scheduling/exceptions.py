"""Exceptions for DAG scheduling and step execution."""

from __future__ import annotations


class StepExecutionError(Exception):
    """Raised when an individual step execution fails after exhausting retries."""

    def __init__(self, step_id: str, message: str) -> None:
        super().__init__(message)
        self.step_id = step_id


class DAGDeadlockError(Exception):
    """Raised when the DAG scheduler cannot make progress due to unresolved dependencies."""
