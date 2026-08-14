from unittest.mock import MagicMock

import pytest

from semantic_search_eng.models.chunk import Chunk
from semantic_search_eng.retrival.retriver_manager import (
    RetriverManager,
)


@pytest.fixture
def mock_settings():
    settings = MagicMock()

    settings.embedding_model = "all-MiniLM-L6-v2"
    settings.embedding_cost_per_million_tokens = 0.0
    settings.default_top_k = 2
    settings.max_top_k = 100

    return settings


@pytest.fixture
def retriver_manager(
    monkeypatch,
    mock_settings,
):
    monkeypatch.setattr(
        "semantic_search_eng.retrival.retriver_manager.get_settings",
        lambda: mock_settings,
    )

    mock_embedding_manager = MagicMock()

    mock_embedding_manager.embed_query.return_value = [
        1.0,
        0.0,
        0.0,
    ]

    monkeypatch.setattr(
        "semantic_search_eng.retrival.retriver_manager.EmbeddingManager",
        lambda: mock_embedding_manager,
    )

    mock_tracker_logger = MagicMock()

    monkeypatch.setattr(
        "semantic_search_eng.retrival.retriver_manager.QueryTrackerLogger",
        lambda: mock_tracker_logger,
    )

    manager = RetriverManager()

    return (
        manager,
        mock_embedding_manager,
        mock_tracker_logger,
    )


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            chunk_id="chunk_0",
            document_id="document_0000",
            tenant_id="conversation_001",
            chunk_index=0,
            text="Highly relevant content.",
            token_count=3,
        ),
        Chunk(
            chunk_id="chunk_1",
            document_id="document_0000",
            tenant_id="conversation_001",
            chunk_index=1,
            text="Unrelated content.",
            token_count=2,
        ),
        Chunk(
            chunk_id="chunk_2",
            document_id="document_0000",
            tenant_id="conversation_001",
            chunk_index=2,
            text="Also relevant content.",
            token_count=3,
        ),
    ]


@pytest.fixture
def sample_embeddings():
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.8, 0.6, 0.0],
    ]


def test_retrieve_returns_top_k(
    retriver_manager,
    sample_chunks,
    sample_embeddings,
):
    manager, _, _ = retriver_manager

    results = manager.retrieve(
        tenant_id="conversation_001",
        query="semantic search",
        chunks=sample_chunks,
        embeddings=sample_embeddings,
        top_k=2,
    )

    assert len(results) == 2


def test_retrieve_orders_by_similarity(
    retriver_manager,
    sample_chunks,
    sample_embeddings,
):
    manager, _, _ = retriver_manager

    results = manager.retrieve(
        tenant_id="conversation_001",
        query="semantic search",
        chunks=sample_chunks,
        embeddings=sample_embeddings,
        top_k=3,
    )

    assert [result.chunk.chunk_id for result in results] == [
        "chunk_0",
        "chunk_2",
        "chunk_1",
    ]


def test_retrieve_returns_expected_similarity_scores(
    retriver_manager,
    sample_chunks,
    sample_embeddings,
):
    manager, _, _ = retriver_manager

    results = manager.retrieve(
        tenant_id="conversation_001",
        query="semantic search",
        chunks=sample_chunks,
        embeddings=sample_embeddings,
        top_k=3,
    )

    assert results[0].similarity_score == pytest.approx(1.0)

    assert results[1].similarity_score == pytest.approx(0.8)

    assert results[2].similarity_score == pytest.approx(0.0)


def test_retrieve_embeds_the_query(
    retriver_manager,
    sample_chunks,
    sample_embeddings,
):
    manager, embedding_manager, _ = retriver_manager

    manager.retrieve(
        tenant_id="conversation_001",
        query="How does search work?",
        chunks=sample_chunks,
        embeddings=sample_embeddings,
        top_k=2,
    )

    embedding_manager.embed_query.assert_called_once_with("How does search work?")


def test_retrieve_empty_chunks_returns_empty(
    retriver_manager,
):
    manager, embedding_manager, tracker_logger = retriver_manager

    result = manager.retrieve(
        tenant_id="conversation_001",
        query="test query",
        chunks=[],
        embeddings=[],
        top_k=5,
    )

    assert result == []

    embedding_manager.embed_query.assert_not_called()
    tracker_logger.track.assert_not_called()


def test_retrieve_rejects_empty_query(
    retriver_manager,
    sample_chunks,
    sample_embeddings,
):
    manager, embedding_manager, _ = retriver_manager

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        manager.retrieve(
            tenant_id="conversation_001",
            query="   ",
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            top_k=2,
        )

    embedding_manager.embed_query.assert_not_called()


def test_retrieve_rejects_mismatched_lengths(
    retriver_manager,
    sample_chunks,
):
    manager, _, _ = retriver_manager

    with pytest.raises(
        ValueError,
        match="Number of chunks must match",
    ):
        manager.retrieve(
            tenant_id="conversation_001",
            query="test query",
            chunks=sample_chunks,
            embeddings=[
                [1.0, 0.0, 0.0],
            ],
            top_k=2,
        )


def test_default_top_k_is_used(
    retriver_manager,
    sample_chunks,
    sample_embeddings,
    mock_settings,
):
    manager, _, _ = retriver_manager

    results = manager.retrieve(
        tenant_id="conversation_001",
        query="semantic search",
        chunks=sample_chunks,
        embeddings=sample_embeddings,
    )

    assert len(results) == mock_settings.default_top_k


def test_top_k_cannot_exceed_maximum(
    retriver_manager,
    sample_chunks,
    sample_embeddings,
    mock_settings,
):
    mock_settings.max_top_k = 2

    manager, _, _ = retriver_manager

    with pytest.raises(
        ValueError,
        match="top_k cannot exceed",
    ):
        manager.retrieve(
            tenant_id="conversation_001",
            query="semantic search",
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            top_k=3,
        )


def test_zero_top_k_is_rejected(
    retriver_manager,
    sample_chunks,
    sample_embeddings,
):
    manager, _, _ = retriver_manager

    with pytest.raises(
        ValueError,
        match="top_k must be greater",
    ):
        manager.retrieve(
            tenant_id="conversation_001",
            query="semantic search",
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            top_k=0,
        )


def test_query_tracker_is_recorded(
    retriver_manager,
    sample_chunks,
    sample_embeddings,
):
    manager, _, tracker_logger = retriver_manager

    manager.retrieve(
        tenant_id="conversation_001",
        query="semantic search",
        chunks=sample_chunks,
        embeddings=sample_embeddings,
        top_k=2,
    )

    tracker_logger.track.assert_called_once()

    tracker = tracker_logger.track.call_args.args[0]

    assert tracker.tenant_id == "conversation_001"
    assert tracker.query == "semantic search"
    assert tracker.embedding_model == ("all-MiniLM-L6-v2")
    assert tracker.query_token_count == 2
    assert tracker.embedding_latency_ms >= 0
    assert tracker.retrieval_latency_ms >= 0
    assert tracker.total_latency_ms >= 0
    assert tracker.estimated_cost == 0.0
    assert tracker.top_k == 2


def test_query_cost_is_zero_for_local_model(
    retriver_manager,
    sample_chunks,
    sample_embeddings,
):
    manager, _, tracker_logger = retriver_manager

    manager.retrieve(
        tenant_id="conversation_001",
        query="semantic search",
        chunks=sample_chunks,
        embeddings=sample_embeddings,
        top_k=2,
    )

    tracker = tracker_logger.track.call_args.args[0]

    assert tracker.estimated_cost == 0.0
