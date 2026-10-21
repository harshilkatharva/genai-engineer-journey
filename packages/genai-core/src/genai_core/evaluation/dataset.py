from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..exceptions.evaluation_exceptions import EvaluationDatasetError
from ..models.evaluation_model import EvalSample, ExpectedAnswer, ExpectedEvidence


class EvalDataset:
    """Evaluation dataset container supporting native and legacy formats."""

    def __init__(self, samples: list[EvalSample], metadata: dict[str, Any] | None = None) -> None:
        self.samples = samples
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[EvalSample]:
        return iter(self.samples)

    def __getitem__(self, index: int) -> EvalSample:
        return self.samples[index]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalDataset:
        """Parse an evaluation dataset from a dictionary."""
        if not isinstance(data, dict):
            raise EvaluationDatasetError("Dataset root must be a JSON object dictionary.")

        raw_samples = data.get("samples") or data.get("questions")
        if raw_samples is None:
            raise EvaluationDatasetError("Dataset must contain a 'samples' or 'questions' key.")

        if not isinstance(raw_samples, list):
            raise EvaluationDatasetError("'samples' or 'questions' must be a list.")

        parsed_samples: list[EvalSample] = []
        for idx, item in enumerate(raw_samples):
            if not isinstance(item, dict):
                raise EvaluationDatasetError(f"Sample at index {idx} must be a dictionary.")

            sample_id = item.get("id", idx + 1)
            query = item.get("query")
            if not query or not isinstance(query, str):
                raise EvaluationDatasetError(
                    f"Sample at index {idx} is missing a valid 'query' string."
                )

            context = item.get("context")
            expected_tools = item.get("expected_tools")

            # Parse expected evidence (supporting native & module-09 formats)
            expected_evidence: ExpectedEvidence | None = None
            evidence_data = item.get("expected_evidence") or item.get("expected")
            if isinstance(evidence_data, dict):
                required_chunks = [
                    str(c).strip() for c in evidence_data.get("required", []) if str(c).strip()
                ]
                raw_any_of = evidence_data.get("any_of", [])
                any_of_groups: list[list[str]] = []
                if raw_any_of:
                    # If flat list of strings: treat as a single alternative group [c1, c2]
                    if all(isinstance(c, (str, int)) for c in raw_any_of):
                        cleaned = [str(c).strip() for c in raw_any_of if str(c).strip()]
                        if cleaned:
                            any_of_groups.append(cleaned)
                    else:
                        for grp in raw_any_of:
                            if isinstance(grp, (list, tuple, set)):
                                cleaned = [str(c).strip() for c in grp if str(c).strip()]
                                if cleaned:
                                    any_of_groups.append(cleaned)
                expected_evidence = ExpectedEvidence(required=required_chunks, any_of=any_of_groups)
            elif "chunk_ids" in item:
                expected_evidence = ExpectedEvidence(
                    required=[str(c).strip() for c in item["chunk_ids"] if str(c).strip()]
                )

            # Parse expected answer
            expected_answer: ExpectedAnswer | None = None
            answer_data = item.get("expected_answer")
            if isinstance(answer_data, dict):
                req_ans = [
                    str(a).strip() for a in answer_data.get("required", []) if str(a).strip()
                ]
                any_ans = [str(a).strip() for a in answer_data.get("any_of", []) if str(a).strip()]
                ref_ans = answer_data.get("reference")
                expected_answer = ExpectedAnswer(
                    required=req_ans, any_of=any_ans, reference=ref_ans
                )
            elif isinstance(answer_data, str):
                expected_answer = ExpectedAnswer(reference=answer_data)

            metadata = {
                k: v
                for k, v in item.items()
                if k
                not in {
                    "id",
                    "query",
                    "context",
                    "expected_evidence",
                    "expected",
                    "expected_answer",
                    "expected_tools",
                    "chunk_ids",
                }
            }

            parsed_samples.append(
                EvalSample(
                    id=sample_id,
                    query=query,
                    context=context,
                    expected_evidence=expected_evidence,
                    expected_answer=expected_answer,
                    expected_tools=expected_tools,
                    metadata=metadata,
                )
            )

        root_metadata = {k: v for k, v in data.items() if k not in {"samples", "questions"}}
        return cls(samples=parsed_samples, metadata=root_metadata)

    @classmethod
    def from_json(cls, path_or_str: str | Path) -> EvalDataset:
        """Load an evaluation dataset from a JSON file path or string."""
        path = Path(path_or_str)
        if path.is_file():
            try:
                content = json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                raise EvaluationDatasetError(
                    f"Failed to read or parse JSON file {path}: {e}"
                ) from e
        else:
            try:
                content = json.loads(str(path_or_str))
            except Exception as e:
                raise EvaluationDatasetError(f"Failed to parse JSON string: {e}") from e

        return cls.from_dict(content)

    @classmethod
    def from_jsonl(cls, path_or_str: str | Path) -> EvalDataset:
        """Load an evaluation dataset from a JSONL file path."""
        path = Path(path_or_str)
        if not path.is_file():
            raise EvaluationDatasetError(f"JSONL file not found at: {path}")

        samples_list: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    item = json.loads(line_str)
                    if not isinstance(item, dict):
                        raise EvaluationDatasetError(f"Line {line_idx} must be a JSON object.")
                    samples_list.append(item)
        except Exception as e:
            raise EvaluationDatasetError(f"Failed to read JSONL file {path}: {e}") from e

        return cls.from_dict({"samples": samples_list})

    def to_dict(self) -> dict[str, Any]:
        """Convert dataset to a dictionary representation."""
        return {
            **self.metadata,
            "samples": [sample.model_dump() for sample in self.samples],
        }

    def save(self, path: str | Path) -> None:
        """Save the dataset to a JSON file."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
