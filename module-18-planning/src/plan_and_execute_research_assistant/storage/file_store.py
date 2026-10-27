"""File-based persistence implementation of RunRecordStore."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from genai_core import safe_log

from plan_and_execute_research_assistant.core import AssistantConfig, get_assistant_config
from plan_and_execute_research_assistant.models import RunRecord
from plan_and_execute_research_assistant.storage.protocol import RunRecordStore

_logger = logging.getLogger("plan_and_execute_research_assistant.storage")


class FileRunRecordStore(RunRecordStore):
    """Persists RunRecords as formatted JSON files in a designated local directory."""

    def __init__(
        self,
        base_dir: Path | str | None = None,
        config: AssistantConfig | None = None,
    ) -> None:
        """Initialize the store and ensure the target directory exists."""
        cfg = config or get_assistant_config()
        self.base_dir = Path(base_dir) if base_dir is not None else cfg.runs_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _run_file_path(self, run_id: str) -> Path:
        """Derive the deterministic filename for a run_id."""
        return self.base_dir / f"run_{run_id}.json"

    def _save_run_sync(self, record: RunRecord) -> None:
        """Synchronously write RunRecord JSON with atomic rename."""
        target_path = self._run_file_path(record.run_id)
        temp_path = self.base_dir / f"run_{record.run_id}.tmp"

        payload = record.model_dump_json(indent=2)
        temp_path.write_text(payload, encoding="utf-8")
        temp_path.replace(target_path)

    def _get_run_sync(self, run_id: str) -> RunRecord | None:
        """Synchronously read and deserialize RunRecord from disk."""
        target_path = self._run_file_path(run_id)
        if not target_path.exists():
            return None

        try:
            content = target_path.read_text(encoding="utf-8")
            return RunRecord.model_validate_json(content)
        except Exception as exc:  # noqa: BLE001
            safe_log(
                _logger,
                logging.ERROR,
                f"Failed to deserialize RunRecord from {target_path}: {exc}",
                run_id=run_id,
                error=str(exc),
            )
            return None

    def _list_runs_sync(self, user_id: str | None = None, limit: int = 50) -> list[RunRecord]:
        """Synchronously scan and load runs from base_dir."""
        records: list[RunRecord] = []
        for file_path in self.base_dir.glob("run_*.json"):
            try:
                content = file_path.read_text(encoding="utf-8")
                rec = RunRecord.model_validate_json(content)
                if user_id is None or rec.user_id == user_id:
                    records.append(rec)
            except Exception as exc:  # noqa: BLE001
                safe_log(
                    _logger,
                    logging.WARNING,
                    f"Skipping corrupted run file {file_path}: {exc}",
                    file_path=str(file_path),
                    error=str(exc),
                )

        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[:limit]

    async def save_run(self, record: RunRecord) -> None:
        """Persist or update a RunRecord asynchronously."""
        await asyncio.to_thread(self._save_run_sync, record)
        safe_log(
            _logger,
            logging.DEBUG,
            f"Saved RunRecord '{record.run_id}' to {self._run_file_path(record.run_id)}",
            run_id=record.run_id,
            status=record.status.value,
        )

    async def get_run(self, run_id: str) -> RunRecord | None:
        """Retrieve a RunRecord by ID asynchronously."""
        return await asyncio.to_thread(self._get_run_sync, run_id)

    async def list_runs(
        self,
        user_id: str | None = None,
        limit: int = 50,
    ) -> list[RunRecord]:
        """List persisted RunRecords asynchronously."""
        return await asyncio.to_thread(self._list_runs_sync, user_id=user_id, limit=limit)
