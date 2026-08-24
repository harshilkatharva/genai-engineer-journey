import json
from pathlib import Path
from typing import Any

from rag_app.core.settings import get_settings


class EvalutionReport:
    def __init__(self):
        self.settings = get_settings()

        self.performance_file = (
            Path(self.settings.log_directory) / "query_performance_tracker.jsonl"
        )

        self.evaluation_dataset_file = Path(self.settings.evalution_dataset)

    def get_report(self):
        performance = self._get_performance()
        report = self._calculate_report(performance)

        version = self.settings.app_version

        evaluation_file = Path(self.settings.evaluation_file_path) / f"evaluation_{version}.json"

        evaluation_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        evaluation_file.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return report

    def _get_performance(self) -> list[dict[str, Any]]:
        if not self.performance_file.exists():
            raise FileNotFoundError(f"Performance tracker not found: {self.performance_file}")

        target_version = self.settings.app_version

        records: list[dict[str, Any]] = []

        with self.performance_file.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)

                if record.get("app_version") == target_version:
                    records.append(record)

        return records

    def _calculate_report(
        self,
        performance: list[dict[str, Any]],
    ) -> dict[str, Any]:
        dataset = json.loads(self.evaluation_dataset_file.read_text(encoding="utf-8"))

        expected_questions = {item["query"]: item for item in dataset["questions"]}

        question_reports = []

        for record in performance:
            query = record["query"]

            evaluation_item = expected_questions.get(query)

            if evaluation_item is None:
                continue

            expected_chunk_ids = set(evaluation_item["chunk_ids"])

            retrieved_chunk_ids = set(record.get("chunk_ids", []))

            matched_chunk_ids = expected_chunk_ids & retrieved_chunk_ids

            expected_count = len(expected_chunk_ids)
            matched_count = len(matched_chunk_ids)

            recall = matched_count / expected_count if expected_count > 0 else 0.0

            question_reports.append(
                {
                    "question_id": evaluation_item["id"],
                    "query": query,
                    "app_version": record["app_version"],
                    "expected_chunk_ids": sorted(expected_chunk_ids),
                    "retrieved_chunk_ids": sorted(retrieved_chunk_ids),
                    "matched_chunk_ids": sorted(matched_chunk_ids),
                    "expected_count": expected_count,
                    "matched_count": matched_count,
                    "recall": recall,
                    "logged_at": record.get("logged_at"),
                }
            )

        recalls = [item["recall"] for item in question_reports]

        average_recall = sum(recalls) / len(recalls) if recalls else 0.0

        return {
            "app_version": self.settings.app_version,
            "metric": "chunk_recall",
            "formula": ("matched_expected_chunk_ids / expected_chunk_ids"),
            "total_questions": len(question_reports),
            "average_recall": average_recall,
            "questions": question_reports,
        }
