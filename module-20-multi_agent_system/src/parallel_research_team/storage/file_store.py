"""File-based atomic checkpoint store implementation."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path

from ..core.config import ResearchTeamConfig, get_research_team_config
from ..exceptions import CheckpointError
from ..models.checkpoint import CheckpointState
from .protocol import CheckpointStore

_logger = logging.getLogger("parallel_research_team.storage.file_store")


class FileCheckpointStore(CheckpointStore):
    """Persists CheckpointState instances as atomic JSON files on disk.

    Uses temporary file creation followed by atomic rename (os.replace) to ensure
    consistent, uncorrupted checkpoints even if interrupted during write.
    """

    def __init__(
        self,
        base_dir: str | Path | None = None,
        config: ResearchTeamConfig | None = None,
    ) -> None:
        cfg = config or get_research_team_config()
        self.base_dir = Path(base_dir or cfg.checkpoint_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _get_path(self, run_id: str) -> Path:
        """Return standardized path for a run's checkpoint file."""
        return self.base_dir / f"checkpoint_{run_id}.json"

    async def save_checkpoint(self, state: CheckpointState) -> None:
        """Atomically persist CheckpointState snapshot to disk."""
        target_path = self._get_path(state.run_id)

        def _write() -> None:
            # Write to temporary file in the same directory for atomic replace
            temp_name: str | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    dir=str(self.base_dir),
                    prefix=f"tmp_cp_{state.run_id}_",
                    suffix=".json",
                    delete=False,
                    encoding="utf-8",
                ) as temp_file:
                    temp_file.write(state.model_dump_json(indent=2))
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                    temp_name = temp_file.name
                os.replace(temp_name, target_path)
            except Exception as exc:
                if temp_name and os.path.exists(temp_name):
                    os.unlink(temp_name)
                raise CheckpointError(
                    f"Failed to persist checkpoint for run '{state.run_id}': {exc}"
                ) from exc

        async with self._lock:
            await asyncio.to_thread(_write)

    async def get_checkpoint(self, run_id: str) -> CheckpointState | None:
        """Load CheckpointState snapshot from disk by run_id."""
        target_path = self._get_path(run_id)
        if not target_path.exists():
            return None

        def _read() -> CheckpointState:
            try:
                content = target_path.read_text(encoding="utf-8")
                return CheckpointState.model_validate_json(content)
            except Exception as exc:
                raise CheckpointError(
                    f"Failed to parse checkpoint file '{target_path}': {exc}"
                ) from exc

        async with self._lock:
            return await asyncio.to_thread(_read)

    async def list_checkpoints(self, limit: int = 50) -> list[str]:
        """List checkpoint run IDs sorted by modification time descending."""

        def _list() -> list[str]:
            files = list(self.base_dir.glob("checkpoint_*.json"))
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            run_ids: list[str] = []
            for f in files[:limit]:
                # Extract run_id from 'checkpoint_<run_id>.json'
                name = f.stem
                if name.startswith("checkpoint_"):
                    run_ids.append(name[len("checkpoint_") :])
            return run_ids

        async with self._lock:
            return await asyncio.to_thread(_list)
