"""Worker protocol defining the interface for bounded research agents."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..core.budget import SharedRunBudget
from ..models import WorkerBrief, WorkerExecutionResult


@runtime_checkable
class ResearchWorkerProtocol(Protocol):
    """Protocol defining the interface for bounded research workers."""

    @property
    def worker_type(self) -> str:
        """Worker type category (e.g. 'default', 'deep_search', 'analyst')."""
        ...

    async def execute(
        self,
        brief: WorkerBrief,
        shared_budget: SharedRunBudget,
    ) -> WorkerExecutionResult:
        """Execute assigned research brief bounded by steps, timeout, and shared budget."""
        ...
