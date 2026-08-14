import json
from datetime import UTC, datetime
from pathlib import Path

from semantic_search_eng.config import get_settings
from semantic_search_eng.models.query_tracker import QueryTracker


class QueryTrackerLogger:
    """
    Logs query/retrieval metrics as JSONL.

    Output:
        user_data/logs/query_tracker.jsonl
    """

    def __init__(self) -> None:
        settings = get_settings()

        self.log_directory = Path(settings.log_directory)
        self.log_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.log_file = self.log_directory / "query_tracker.jsonl"

    def track(
        self,
        tracker: QueryTracker,
    ) -> None:
        record = self._build_record(tracker)

        self._write(record)

    def _build_record(
        self,
        tracker: QueryTracker,
    ) -> dict:
        record = tracker.model_dump()

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
