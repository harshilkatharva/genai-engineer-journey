from __future__ import annotations

import json
from pathlib import Path

import pytest

from genai_core.evaluation import EvalDataset
from genai_core.exceptions import EvaluationDatasetError


def test_dataset_from_dict_native() -> None:
    data = {
        "name": "test_dataset",
        "samples": [
            {
                "id": 1,
                "query": "What is the capital of France?",
                "expected_answer": {"required": ["Paris"], "any_of": []},
                "expected_evidence": {"required": ["doc_1"], "any_of": [["doc_2", "doc_3"]]},
            }
        ],
    }

    dataset = EvalDataset.from_dict(data)
    assert len(dataset) == 1
    assert dataset.metadata["name"] == "test_dataset"
    sample = dataset[0]
    assert sample.id == 1
    assert sample.query == "What is the capital of France?"
    assert sample.expected_answer is not None
    assert sample.expected_answer.required == ["Paris"]
    assert sample.expected_evidence is not None
    assert sample.expected_evidence.required == ["doc_1"]
    assert sample.expected_evidence.any_of == [["doc_2", "doc_3"]]


def test_dataset_from_dict_legacy_module_09_format() -> None:
    # Mimics module-09-rag/evalution_results/evalution_dataset.json
    legacy_data = {
        "version": "0.1.0",
        "metric": {"name": "chunk_recall"},
        "questions": [
            {
                "id": 101,
                "query": "What was the name of the ship?",
                "expected_answer": {"required": ["Pharaon"], "any_of": []},
                "expected": {
                    "required": [],
                    "any_of": ["chunk_0002", "chunk_0003"],
                },
            },
            {
                "id": 102,
                "query": "How Captain died?",
                "expected_answer": {"required": ["brain fever"]},
                "expected": {
                    "required": ["chunk_0005"],
                    "any_of": [],
                },
            },
        ],
    }

    dataset = EvalDataset.from_dict(legacy_data)
    assert len(dataset) == 2
    s1 = dataset[0]
    assert s1.id == 101
    assert s1.expected_evidence is not None
    assert s1.expected_evidence.required == []
    # Flat any_of string list in legacy format is normalized to a single group
    assert s1.expected_evidence.any_of == [["chunk_0002", "chunk_0003"]]

    s2 = dataset[1]
    assert s2.id == 102
    assert s2.expected_evidence is not None
    assert s2.expected_evidence.required == ["chunk_0005"]


def test_dataset_from_json_and_save(tmp_path: Path) -> None:
    data = {
        "samples": [
            {"id": "q1", "query": "Query 1", "expected_answer": {"reference": "Ref 1"}},
            {"id": "q2", "query": "Query 2", "chunk_ids": ["c1", "c2"]},
        ]
    }
    file_path = tmp_path / "dataset.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    dataset = EvalDataset.from_json(file_path)
    assert len(dataset) == 2
    assert dataset[0].id == "q1"
    assert dataset[0].expected_answer is not None
    assert dataset[0].expected_answer.reference == "Ref 1"
    assert dataset[1].expected_evidence is not None
    assert dataset[1].expected_evidence.required == ["c1", "c2"]

    # Test save
    saved_path = tmp_path / "saved.json"
    dataset.save(saved_path)
    assert saved_path.exists()
    reloaded = EvalDataset.from_json(saved_path)
    assert len(reloaded) == 2


def test_dataset_from_jsonl(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "dataset.jsonl"
    jsonl_path.write_text(
        '{"id": "1", "query": "Q1"}\n{"id": "2", "query": "Q2"}\n',
        encoding="utf-8",
    )

    dataset = EvalDataset.from_jsonl(jsonl_path)
    assert len(dataset) == 2
    assert dataset[0].query == "Q1"
    assert dataset[1].query == "Q2"


def test_dataset_invalid_format_raises() -> None:
    with pytest.raises(EvaluationDatasetError, match="must be a JSON object dictionary"):
        EvalDataset.from_dict("not-a-dict")  # type: ignore[arg-type]

    with pytest.raises(EvaluationDatasetError, match="must contain a 'samples' or 'questions' key"):
        EvalDataset.from_dict({"something_else": []})

    with pytest.raises(EvaluationDatasetError, match="missing a valid 'query' string"):
        EvalDataset.from_dict({"samples": [{"id": 1}]})
