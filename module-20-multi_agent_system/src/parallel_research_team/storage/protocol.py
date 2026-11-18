"""Protocol definition for run checkpoint persistence."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models.checkpoint import CheckpointState


@runtime_checkable
class CheckpointStore(Protocol):
    """Protocol for persisting, retrieving, and listing research run checkpoints."""

    async def save_checkpoint(self, state: CheckpointState) -> None:
        """Persist or update a research run CheckpointState.

        Args:
            state: The CheckpointState snapshot to persist.
        """
        ...

    async def get_checkpoint(self, run_id: str) -> CheckpointState | None:
        """Retrieve a CheckpointState by its run_id.

        Args:
            run_id: Unique UUID of the run.

        Returns:
            The loaded CheckpointState, or None if not found.
        """
        ...

    async def list_checkpoints(self, limit: int = 50) -> list[str]:
        """List persisted checkpoint run IDs, sorted by last modified descending.

        Args:
            limit: Maximum number of run IDs to return.

        Returns:
            List of run_id strings.
        """
        ...
