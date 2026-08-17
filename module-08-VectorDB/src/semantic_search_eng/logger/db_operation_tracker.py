import json
from datetime import UTC, datetime
from pathlib import Path

from semantic_search_eng.config import get_settings
from semantic_search_eng.models.index_batch_tracker import IndexBatchTracker
from semantic_search_eng.models.db_query_tracker import DBQueryTracker


class DBOperationTracker:
    """
    Logs database operations as JSONL.

    Tracks:
        - Index batch insertions
        - Query executions

    Output:
        user_data/logs/db_operation_tracker.jsonl
    """

    def __init__(self) -> None:
        settings = get_settings()

        self.log_directory = Path(settings.log_directory)
        self.log_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.log_file = self.log_directory / "db_operation_tracker.jsonl"

    def track_index_batch(
        self,
        tracker: IndexBatchTracker,
    ) -> None:
        record = self._build_record(tracker, operation_type="index_batch")
        self._write(record)

    def track_query(
        self,
        tracker: DBQueryTracker,
    ) -> None:
        record = self._build_record(tracker, operation_type="query")
        self._write(record)

    def _build_record(
        self,
        tracker,
        operation_type: str,
    ) -> dict:
        record = tracker.model_dump()
        record["operation_type"] = operation_type
        record["logged_at"] = datetime.now(UTC).isoformat()

        return record

    def _write(
        self,
        record: dict,
    ) -> None:
        with self.log_file.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )
            file.write("\n")
