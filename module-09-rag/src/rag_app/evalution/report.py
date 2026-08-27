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
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return report

    # ============================================================
    # PERFORMANCE
    # ============================================================

    def _get_performance(self) -> list[dict[str, Any]]:
        if not self.performance_file.exists():
            raise FileNotFoundError(f"Performance tracker not found: {self.performance_file}")

        target_version = self.settings.app_version

        records: list[dict[str, Any]] = []

        with self.performance_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)

                if record.get("app_version") == target_version:
                    records.append(record)

        return records

    # ============================================================
    # NORMALIZATION
    # ============================================================

    def _normalize_chunk_id(self, chunk_id: Any) -> str | None:
        """
        Normalize a single chunk ID.

        Handles accidental whitespace:

            " chunk_123 " -> "chunk_123"
        """

        if not isinstance(chunk_id, str):
            return None

        chunk_id = chunk_id.strip()

        if not chunk_id:
            return None

        return chunk_id

    def _normalize_chunk_ids(
        self,
        chunk_ids: Any,
    ) -> set[str]:
        """
        Normalize a collection of chunk IDs.

        IMPORTANT:
        A string is treated as ONE chunk ID.

        This prevents:

            "chunk_123"

        from becoming:

            {"c", "h", "u", ...}
        """

        if chunk_ids is None:
            return set()

        # A single chunk ID
        if isinstance(chunk_ids, str):
            normalized = self._normalize_chunk_id(chunk_ids)

            return {normalized} if normalized else set()

        # Multiple chunk IDs
        if isinstance(chunk_ids, (list, tuple, set)):
            result: set[str] = set()

            for chunk_id in chunk_ids:
                normalized = self._normalize_chunk_id(chunk_id)

                if normalized:
                    result.add(normalized)

            return result

        return set()

    # ============================================================
    # ANY-OF NORMALIZATION
    # ============================================================

    def _normalize_any_of(
        self,
        any_of: Any,
    ) -> list[set[str]]:
        if not any_of:
            return []

        if all(isinstance(item, str) for item in any_of):
            group = self._normalize_chunk_ids(any_of)

            return [group] if group else []

        groups: list[set[str]] = []

        for item in any_of:
            group = self._normalize_chunk_ids(item)

            if group:
                groups.append(group)

        return groups

    # ============================================================
    # QUESTION EVALUATION
    # ============================================================

    def _calculate_question_recall(
        self,
        evaluation_item: dict[str, Any],
        retrieved_chunk_ids: set[str],
    ) -> dict[str, Any]:
        expected = evaluation_item.get("expected")

        if expected is None:
            required_chunk_ids = self._normalize_chunk_ids(evaluation_item.get("chunk_ids", []))

            matched_chunk_ids = required_chunk_ids & retrieved_chunk_ids

            missing_chunk_ids = required_chunk_ids - retrieved_chunk_ids

            expected_count = len(required_chunk_ids)
            matched_count = len(matched_chunk_ids)

            recall = matched_count / expected_count if expected_count > 0 else 0.0

            return {
                "evaluation_type": "required",
                "expected_chunk_ids": sorted(required_chunk_ids),
                "required_chunk_ids": sorted(required_chunk_ids),
                "matched_required_chunk_ids": sorted(matched_chunk_ids),
                "missing_required_chunk_ids": sorted(missing_chunk_ids),
                "any_of": [],
                "retrieved_chunk_ids": sorted(retrieved_chunk_ids),
                "matched_chunk_ids": sorted(matched_chunk_ids),
                "unexpected_chunk_ids": sorted(retrieved_chunk_ids - required_chunk_ids),
                "expected_count": expected_count,
                "matched_count": matched_count,
                "recall": recall,
            }

        required_chunk_ids = self._normalize_chunk_ids(expected.get("required", []))

        any_of_groups = self._normalize_any_of(expected.get("any_of", []))

        # ========================================================
        # REQUIRED CHUNKS
        # ========================================================

        matched_required = required_chunk_ids & retrieved_chunk_ids

        missing_required = required_chunk_ids - retrieved_chunk_ids

        # ========================================================
        # ANY-OF GROUPS
        # ========================================================

        any_of_results = []

        for index, alternatives in enumerate(
            any_of_groups,
            start=1,
        ):
            matched = alternatives & retrieved_chunk_ids

            satisfied = bool(matched)

            any_of_results.append(
                {
                    "group": index,
                    "expected_chunk_ids": sorted(alternatives),
                    "matched_chunk_ids": sorted(matched),
                    "missing_chunk_ids": sorted(alternatives - retrieved_chunk_ids),
                    "satisfied": satisfied,
                }
            )

        total_evidence_requirements = len(required_chunk_ids) + len(any_of_groups)

        satisfied_required_count = len(matched_required)

        satisfied_any_of_count = sum(1 for group in any_of_results if group["satisfied"])

        satisfied_evidence_requirements = satisfied_required_count + satisfied_any_of_count

        recall = (
            satisfied_evidence_requirements / total_evidence_requirements
            if total_evidence_requirements > 0
            else 0.0
        )

        # ========================================================
        # REPORTING IDS
        # ========================================================

        all_expected_chunk_ids = set(required_chunk_ids)

        for group in any_of_groups:
            all_expected_chunk_ids.update(group)

        matched_chunk_ids = all_expected_chunk_ids & retrieved_chunk_ids

        unexpected_chunk_ids = retrieved_chunk_ids - all_expected_chunk_ids

        return {
            "evaluation_type": "required_and_any_of",
            "expected_chunk_ids": sorted(all_expected_chunk_ids),
            "required_chunk_ids": sorted(required_chunk_ids),
            "matched_required_chunk_ids": sorted(matched_required),
            "missing_required_chunk_ids": sorted(missing_required),
            "any_of": any_of_results,
            "retrieved_chunk_ids": sorted(retrieved_chunk_ids),
            "matched_chunk_ids": sorted(matched_chunk_ids),
            "unexpected_chunk_ids": sorted(unexpected_chunk_ids),
            "required_count": len(required_chunk_ids),
            "matched_required_count": len(matched_required),
            "any_of_group_count": len(any_of_groups),
            "satisfied_any_of_count": (satisfied_any_of_count),
            "total_evidence_requirements": (total_evidence_requirements),
            "satisfied_evidence_requirements": (satisfied_evidence_requirements),
            "recall": recall,
        }

    # ============================================================
    # FULL REPORT
    # ============================================================

    def _calculate_report(
        self,
        performance: list[dict[str, Any]],
    ) -> dict[str, Any]:
        dataset = json.loads(self.evaluation_dataset_file.read_text(encoding="utf-8"))

        expected_questions = {item["query"]: item for item in dataset["questions"]}

        question_reports = []

        for record in performance:
            query = record.get("query")

            evaluation_item = expected_questions.get(query)

            if evaluation_item is None:
                continue

            retrieved_chunk_ids = self._normalize_chunk_ids(record.get("chunk_ids", []))

            question_result = self._calculate_question_recall(
                evaluation_item=evaluation_item,
                retrieved_chunk_ids=retrieved_chunk_ids,
            )

            question_reports.append(
                {
                    "question_id": evaluation_item["id"],
                    "query": query,
                    "app_version": record["app_version"],
                    **question_result,
                    "logged_at": record.get("logged_at"),
                }
            )

        # ========================================================
        # OVERALL METRICS
        # ========================================================

        recalls = [item["recall"] for item in question_reports]

        average_recall = sum(recalls) / len(recalls) if recalls else 0.0

        perfect_recall_questions = sum(1 for item in question_reports if item["recall"] == 1.0)

        zero_recall_questions = sum(1 for item in question_reports if item["recall"] == 0.0)

        return {
            "app_version": self.settings.app_version,
            "metric": "evidence_recall",
            "formula": ("satisfied_evidence_requirements / total_evidence_requirements"),
            "total_questions": len(question_reports),
            "average_recall": average_recall,
            "perfect_recall_questions": (perfect_recall_questions),
            "zero_recall_questions": (zero_recall_questions),
            "questions": question_reports,
        }
