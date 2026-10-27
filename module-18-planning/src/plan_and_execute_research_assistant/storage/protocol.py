"""RunRecordStore protocol interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from plan_and_execute_research_assistant.models import RunRecord


@runtime_checkable
class RunRecordStore(Protocol):
    """Protocol for persisting, retrieving, and listing assistant RunRecords."""

    async def save_run(self, record: RunRecord) -> None:
        """Persist or update a RunRecord.

        Args:
            record: The RunRecord to persist.
        """
        ...

    async def get_run(self, run_id: str) -> RunRecord | None:
        """Retrieve a RunRecord by its run_id.

        Args:
            run_id: Unique UUID of the run.

        Returns:
            The loaded RunRecord, or None if not found.
        """
        ...

    async def list_runs(
        self,
        user_id: str | None = None,
        limit: int = 50,
    ) -> list[RunRecord]:
        """List persisted RunRecords, optionally filtered by user_id.

        Args:
            user_id: Optional user filter.
            limit: Maximum records to return.

        Returns:
            List of RunRecords, sorted by created_at descending.
        """
        ...
