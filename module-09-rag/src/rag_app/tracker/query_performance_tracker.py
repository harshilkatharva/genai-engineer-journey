import json
from datetime import UTC, datetime
from pathlib import Path

from rag_app.core import get_settings
from rag_app.models.tracker.query_performance_tracker import (
    QueryPerformanceTracker,
)


class QueryPerformanceTrackerLogger:
    """
    Logs RAG query performance as JSONL.

    Tracks:
        - Application version
        - User query
        - Number of queries
        - Retrieved chunk IDs
        - LLM answer
        - Query latency
        - Retrieval latency
        - Prompt latency
        - LLM latency
        - Total latency

    Output:
        user_data/logs/query_performance_tracker.jsonl
    """

    def __init__(self) -> None:
        settings = get_settings()

        self.log_directory = Path(settings.log_directory)
        self.log_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.log_file = self.log_directory / "query_performance_tracker.jsonl"

    def track(
        self,
        tracker: QueryPerformanceTracker,
    ) -> None:
        record = self._build_record(tracker)
        self._write(record)

    def _build_record(
        self,
        tracker: QueryPerformanceTracker,
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
