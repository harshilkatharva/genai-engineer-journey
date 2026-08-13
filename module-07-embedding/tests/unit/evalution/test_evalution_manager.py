from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from semantic_search_eng.models.chunk import Chunk
from semantic_search_eng.models.search_response import (
    SearchResult,
)


@pytest.fixture
def evaluation_file(tmp_path) -> Path:
    path = tmp_path / "evalution.json"

    cases = [
        {
            "query": "What is semantic search?",
            "relevant_chunk_ids": [
                "chunk_0",
            ],
        },
        {
            "query": "How does embeddings work?",
            "relevant_chunk_ids": [
                "chunk_1",
            ],
        },
        {
            "query": "How does retrieval work?",
            "relevant_chunk_ids": [
                "chunk_2",
            ],
        },
    ]

    path.write_text(
        json.dumps(
            cases,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            chunk_id="chunk_0",
            document_id="document_0000",
            conversation_id="conversation_001",
            chunk_index=0,
            text="Semantic search finds information by meaning.",
            token_count=7,
        ),
        Chunk(
            chunk_id="chunk_1",
            document_id="document_0000",
            conversation_id="conversation_001",
            chunk_index=1,
            text="Embeddings convert text into vectors.",
            token_count=6,
        ),
        Chunk(
            chunk_id="chunk_2",
            document_id="document_0000",
            conversation_id="conversation_001",
            chunk_index=2,
            text="Retrieval ranks chunks by similarity.",
            token_count=6,
        ),
    ]


@pytest.fixture
def evaluation_manager(
    monkeypatch,
    evaluation_file,
    sample_chunks,
):
    mock_settings = MagicMock()
    mock_settings.evaluation_file = str(evaluation_file)
    mock_settings.data_directory = str(evaluation_file.parent / "data")
    mock_settings.embedding_model = "all-MiniLM-L6-v2"
    mock_settings.embedding_cost_per_million_tokens = 0.0
    mock_settings.default_top_k = 5
    mock_settings.max_top_k = 100

    monkeypatch.setattr(
        "semantic_search_eng.config.settings.get_settings",
        lambda: mock_settings,
    )

    from semantic_search_eng.retrival import (
        retriver_manager as retriver_module,
    )

    mock_retriver = MagicMock()

    monkeypatch.setattr(
        retriver_module,
        "RetriverManager",
        lambda: mock_retriver,
    )

    from semantic_search_eng.evaluation.evalution_manager import (
        EvalutionManager,
    )

    manager = EvalutionManager(evaluation_file=str(evaluation_file))

    manager.data_manager = MagicMock()

    manager.data_manager.get_embeddings.return_value = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    chunks_directory = evaluation_file.parent / "data"

    conversation_directory = chunks_directory / "conversation_001" / "chunks"

    conversation_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunk_file = conversation_directory / "document_0000.json"

    chunk_file.write_text(
        json.dumps([chunk.model_dump() for chunk in sample_chunks]),
        encoding="utf-8",
    )

    manager.data_manager.base_path = chunks_directory

    manager.retriver_manager = mock_retriver

    return manager, mock_retriver


def test_load_cases(
    evaluation_manager,
):
    manager, _ = evaluation_manager

    cases = manager.load_cases()

    assert len(cases) == 3

    assert cases[0]["query"] == ("What is semantic search?")

    assert cases[0]["relevant_chunk_ids"] == ["chunk_0"]


def test_load_cases_raises_for_missing_file(
    tmp_path,
):
    from semantic_search_eng.evaluation.evalution_manager import (
        EvalutionManager,
    )

    manager = EvalutionManager(evaluation_file=str(tmp_path / "missing.json"))

    with pytest.raises(
        FileNotFoundError,
        match="Evaluation file not found",
    ):
        manager.load_cases()


def test_evaluate_returns_expected_recall(
    evaluation_manager,
    sample_chunks,
):
    manager, mock_retriver = evaluation_manager

    mock_retriver.retrieve.side_effect = [
        [
            SearchResult(
                chunk=sample_chunks[0],
                similarity_score=0.95,
            ),
        ],
        [
            SearchResult(
                chunk=sample_chunks[1],
                similarity_score=0.91,
            ),
        ],
        [
            SearchResult(
                chunk=sample_chunks[2],
                similarity_score=0.88,
            ),
        ],
    ]

    result = manager.evaluate(
        conversation_id="conversation_001",
        top_k=1,
    )

    assert result["conversation_id"] == ("conversation_001")

    assert result["top_k"] == 1
    assert result["total_queries"] == 3
    assert result["successful_queries"] == 3
    assert result["recall_at_k"] == 1.0

    assert len(result["results"]) == 3


def test_evaluate_partial_recall(
    evaluation_manager,
    sample_chunks,
):
    manager, mock_retriver = evaluation_manager

    mock_retriver.retrieve.side_effect = [
        [
            SearchResult(
                chunk=sample_chunks[0],
                similarity_score=0.95,
            ),
        ],
        [
            SearchResult(
                chunk=sample_chunks[0],
                similarity_score=0.91,
            ),
        ],
        [
            SearchResult(
                chunk=sample_chunks[2],
                similarity_score=0.88,
            ),
        ],
    ]

    result = manager.evaluate(
        conversation_id="conversation_001",
        top_k=1,
    )

    assert result["total_queries"] == 3
    assert result["successful_queries"] == 2
    assert result["recall_at_k"] == pytest.approx(2 / 3)


def test_evaluate_records_matched_chunk_ids(
    evaluation_manager,
    sample_chunks,
):
    manager, mock_retriver = evaluation_manager

    mock_retriver.retrieve.side_effect = [
        [
            SearchResult(
                chunk=sample_chunks[0],
                similarity_score=0.95,
            ),
        ],
        [
            SearchResult(
                chunk=sample_chunks[1],
                similarity_score=0.90,
            ),
        ],
        [
            SearchResult(
                chunk=sample_chunks[2],
                similarity_score=0.85,
            ),
        ],
    ]

    result = manager.evaluate(
        conversation_id="conversation_001",
        top_k=1,
    )

    first_result = result["results"][0]

    assert first_result["relevant_chunk_ids"] == ["chunk_0"]

    assert first_result["retrieved_chunk_ids"] == ["chunk_0"]

    assert first_result["matched_chunk_ids"] == ["chunk_0"]

    assert first_result["hit_at_k"] is True


def test_evaluate_records_missed_query(
    evaluation_manager,
    sample_chunks,
):
    manager, mock_retriver = evaluation_manager

    mock_retriver.retrieve.side_effect = [
        [
            SearchResult(
                chunk=sample_chunks[1],
                similarity_score=0.95,
            ),
        ],
        [
            SearchResult(
                chunk=sample_chunks[1],
                similarity_score=0.91,
            ),
        ],
        [
            SearchResult(
                chunk=sample_chunks[1],
                similarity_score=0.88,
            ),
        ],
    ]

    result = manager.evaluate(
        conversation_id="conversation_001",
        top_k=1,
    )

    assert result["successful_queries"] == 1
    assert result["recall_at_k"] == pytest.approx(1 / 3)

    first_result = result["results"][0]

    assert first_result["hit_at_k"] is False
    assert first_result["matched_chunk_ids"] == []


def test_evaluate_calls_retriever_for_every_query(
    evaluation_manager,
    sample_chunks,
):
    manager, mock_retriver = evaluation_manager

    mock_retriver.retrieve.side_effect = [
        [],
        [],
        [],
    ]

    manager.evaluate(
        conversation_id="conversation_001",
        top_k=5,
    )

    assert mock_retriver.retrieve.call_count == 3


def test_evaluate_uses_requested_top_k(
    evaluation_manager,
):
    manager, mock_retriver = evaluation_manager

    mock_retriver.retrieve.side_effect = [
        [],
        [],
        [],
    ]

    manager.evaluate(
        conversation_id="conversation_001",
        top_k=7,
    )

    for call in mock_retriver.retrieve.call_args_list:
        assert call.kwargs["top_k"] == 7


def test_evaluate_handles_empty_evaluation_set(
    monkeypatch,
    tmp_path,
):
    evaluation_path = tmp_path / "empty_evalution.json"

    evaluation_path.write_text(
        "[]",
        encoding="utf-8",
    )

    mock_settings = MagicMock()
    mock_settings.evaluation_file = str(evaluation_path)
    mock_settings.data_directory = str(tmp_path / "data")

    monkeypatch.setattr(
        "semantic_search_eng.config.settings.get_settings",
        lambda: mock_settings,
    )

    from semantic_search_eng.evaluation.evalution_manager import (
        EvalutionManager,
    )

    manager = EvalutionManager(evaluation_file=str(evaluation_path))

    manager.data_manager = MagicMock()
    manager.retriver_manager = MagicMock()

    result = manager.evaluate(
        conversation_id="conversation_001",
        top_k=5,
    )

    assert result["total_queries"] == 0
    assert result["successful_queries"] == 0
    assert result["recall_at_k"] == 0.0
    assert result["results"] == []
