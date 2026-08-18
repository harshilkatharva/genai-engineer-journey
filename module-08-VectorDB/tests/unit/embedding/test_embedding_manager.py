from unittest.mock import MagicMock
from uuid import UUID

import numpy as np
import pytest

from semantic_search_eng.embedding.embedding_manager import (
    EmbeddingManager,
)
from semantic_search_eng.models.chunk import Chunk


@pytest.fixture
def mock_settings():
    settings = MagicMock()

    settings.embedding_model = "all-MiniLM-L6-v2"
    settings.embedding_batch_size = 2
    settings.embedding_cost_per_million_tokens = 0.0
    settings.data_directory = "user_data/data"
    settings.log_directory = "user_data/logs"

    return settings


@pytest.fixture
def mock_model():
    model = MagicMock()

    def encode(
        texts,
        batch_size=None,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ):
        if isinstance(texts, str):
            return np.array(
                [0.6, 0.8, 0.0],
                dtype=np.float32,
            )

        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.6, 0.8, 0.0],
            ],
            dtype=np.float32,
        )[: len(texts)]

    model.encode.side_effect = encode

    return model


@pytest.fixture
def embedding_manager(
    monkeypatch,
    mock_settings,
    mock_model,
):
    monkeypatch.setattr(
        "semantic_search_eng.embedding.embedding_manager.get_settings",
        lambda: mock_settings,
    )

    monkeypatch.setattr(
        "semantic_search_eng.embedding.embedding_manager.SentenceTransformer",
        lambda model_name: mock_model,
    )

    mock_data_manager = MagicMock()

    monkeypatch.setattr(
        "semantic_search_eng.embedding.embedding_manager.DataManager",
        lambda: mock_data_manager,
    )

    mock_tracker_logger = MagicMock()

    monkeypatch.setattr(
        "semantic_search_eng.embedding.embedding_manager.EmbeddingTrackerLogger",
        lambda: mock_tracker_logger,
    )

    manager = EmbeddingManager()

    return (
        manager,
        mock_model,
        mock_data_manager,
        mock_tracker_logger,
    )


@pytest.fixture
def tenant_id():
    return UUID("11111111-1111-1111-1111-111111111111")


@pytest.fixture
def document_id():
    return UUID("22222222-2222-2222-2222-222222222222")


@pytest.fixture
def sample_chunks(
    tenant_id,
    document_id,
):
    return [
        Chunk(
            chunk_id="document_0000_chunk_0000",
            document_id=document_id,
            tenant_id=tenant_id,
            document_type="text",
            metadata={},
            chunk_index=0,
            text="First chunk.",
            token_count=2,
        ),
        Chunk(
            chunk_id="document_0000_chunk_0001",
            document_id=document_id,
            tenant_id=tenant_id,
            document_type="text",
            metadata={},
            chunk_index=1,
            text="Second chunk.",
            token_count=2,
        ),
        Chunk(
            chunk_id="document_0000_chunk_0002",
            document_id=document_id,
            tenant_id=tenant_id,
            document_type="text",
            metadata={},
            chunk_index=2,
            text="Third chunk.",
            token_count=2,
        ),
    ]


def test_model_is_loaded_with_configured_model(
    embedding_manager,
    mock_model,
):
    manager, _, _, _ = embedding_manager

    assert manager.model is mock_model


def test_embed_chunks_returns_one_embedding_per_chunk(
    embedding_manager,
    sample_chunks,
    tenant_id,
):
    manager, _, _, _ = embedding_manager

    embeddings = manager.embed_chunks(
        tenant_id=tenant_id,
        chunks=sample_chunks,
    )

    assert len(embeddings) == len(sample_chunks)

    assert embeddings[0] == pytest.approx(
        [1.0, 0.0, 0.0],
    )

    assert embeddings[1] == pytest.approx(
        [0.0, 1.0, 0.0],
    )

    assert embeddings[2] == pytest.approx(
        [0.6, 0.8, 0.0],
    )


def test_embed_chunks_uses_configured_batch_size(
    embedding_manager,
    sample_chunks,
    mock_settings,
    tenant_id,
):
    manager, model, _, _ = embedding_manager

    manager.embed_chunks(
        tenant_id=tenant_id,
        chunks=sample_chunks,
    )

    call_kwargs = model.encode.call_args.kwargs

    assert call_kwargs["batch_size"] == (mock_settings.embedding_batch_size)


def test_embed_chunks_uses_normalized_embeddings(
    embedding_manager,
    sample_chunks,
    tenant_id,
):
    manager, model, _, _ = embedding_manager

    manager.embed_chunks(
        tenant_id=tenant_id,
        chunks=sample_chunks,
    )

    call_kwargs = model.encode.call_args.kwargs

    assert call_kwargs["normalize_embeddings"] is True


def test_embed_chunks_uses_expected_encode_arguments(
    embedding_manager,
    sample_chunks,
    mock_settings,
    tenant_id,
):
    manager, model, _, _ = embedding_manager

    manager.embed_chunks(
        tenant_id=tenant_id,
        chunks=sample_chunks,
    )

    model.encode.assert_called_once()

    args = model.encode.call_args.args
    kwargs = model.encode.call_args.kwargs

    assert args[0] == [
        "First chunk.",
        "Second chunk.",
        "Third chunk.",
    ]

    assert kwargs["batch_size"] == (mock_settings.embedding_batch_size)

    assert kwargs["show_progress_bar"] is True
    assert kwargs["convert_to_numpy"] is True
    assert kwargs["normalize_embeddings"] is True


def test_embed_chunks_returns_empty_for_empty_input(
    embedding_manager,
    tenant_id,
):
    manager, model, data_manager, tracker_logger = embedding_manager

    result = manager.embed_chunks(
        tenant_id=tenant_id,
        chunks=[],
    )

    assert result == []

    model.encode.assert_not_called()
    data_manager.save_embeddings.assert_not_called()
    tracker_logger.track.assert_not_called()


def test_embedding_tracker_is_recorded(
    embedding_manager,
    sample_chunks,
    tenant_id,
):
    manager, _, _, tracker_logger = embedding_manager

    manager.embed_chunks(
        tenant_id=tenant_id,
        chunks=sample_chunks,
    )

    tracker_logger.track.assert_called_once()

    tracker = tracker_logger.track.call_args.args[0]

    assert tracker.tenant_id == str(tenant_id)
    assert tracker.embedding_model == "all-MiniLM-L6-v2"
    assert tracker.total_chunks == 3
    assert tracker.total_tokens == 6
    assert tracker.latency_ms >= 0
    assert tracker.estimated_cost == 0.0


def test_embed_query_returns_single_vector(
    embedding_manager,
):
    manager, model, _, _ = embedding_manager

    result = manager.embed_query(
        "How does semantic search work?",
    )

    assert result == pytest.approx(
        [0.6, 0.8, 0.0],
    )

    model.encode.assert_called_once_with(
        "How does semantic search work?",
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def test_embed_query_rejects_empty_query(
    embedding_manager,
):
    manager, model, _, _ = embedding_manager

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        manager.embed_query("   ")

    model.encode.assert_not_called()


def test_embed_documents_groups_results_by_document(
    embedding_manager,
    sample_chunks,
    tenant_id,
):
    manager, _, data_manager, _ = embedding_manager

    second_document_id = UUID(
        "33333333-3333-3333-3333-333333333333",
    )

    second_document_chunks = [
        Chunk(
            chunk_id="document_0001_chunk_0000",
            document_id=second_document_id,
            tenant_id=tenant_id,
            document_type="text",
            metadata={},
            chunk_index=0,
            text="Another chunk.",
            token_count=2,
        ),
    ]

    chunks_by_document = {
        "document_0000": sample_chunks[:2],
        "document_0001": second_document_chunks,
    }

    result = manager.embed_documents(
        tenant_id=tenant_id,
        documents=chunks_by_document,
    )

    assert set(result.keys()) == {
        "document_0000",
        "document_0001",
    }

    assert len(result["document_0000"]) == 2
    assert len(result["document_0001"]) == 1

    data_manager.save_embeddings.assert_not_called()


def test_embed_documents_calls_embed_chunks_for_each_document(
    embedding_manager,
    sample_chunks,
    tenant_id,
):
    manager, _, _, _ = embedding_manager

    second_document_id = UUID(
        "33333333-3333-3333-3333-333333333333",
    )

    second_document_chunks = [
        Chunk(
            chunk_id="document_0001_chunk_0000",
            document_id=second_document_id,
            tenant_id=tenant_id,
            document_type="text",
            metadata={},
            chunk_index=0,
            text="Another chunk.",
            token_count=2,
        ),
    ]

    documents = {
        "document_0000": sample_chunks[:2],
        "document_0001": second_document_chunks,
    }

    manager.embed_chunks = MagicMock(
        side_effect=[
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            [
                [0.6, 0.8, 0.0],
            ],
        ],
    )

    result = manager.embed_documents(
        tenant_id=tenant_id,
        documents=documents,
    )

    assert result == {
        "document_0000": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        "document_0001": [
            [0.6, 0.8, 0.0],
        ],
    }

    assert manager.embed_chunks.call_count == 2


def test_embed_chunks_calculates_local_embedding_cost(
    embedding_manager,
    sample_chunks,
    mock_settings,
    tenant_id,
):
    mock_settings.embedding_cost_per_million_tokens = 2.0

    manager, _, _, tracker_logger = embedding_manager

    manager.embed_chunks(
        tenant_id=tenant_id,
        chunks=sample_chunks,
    )

    tracker = tracker_logger.track.call_args.args[0]

    expected_cost = (6 / 1_000_000) * 2.0

    assert tracker.estimated_cost == pytest.approx(
        expected_cost,
    )


def test_calculate_cost(
    embedding_manager,
    mock_settings,
):
    manager, _, _, _ = embedding_manager

    mock_settings.embedding_cost_per_million_tokens = 10.0

    result = manager._calculate_cost(500_000)

    assert result == pytest.approx(5.0)


def test_save_embeddings_groups_embeddings_by_document(
    embedding_manager,
    sample_chunks,
    tenant_id,
):
    manager, _, data_manager, _ = embedding_manager

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.6, 0.8, 0.0],
    ]

    manager._save_embeddings(
        tenant_id=tenant_id,
        chunks=sample_chunks,
        embeddings=embeddings,
    )

    data_manager.save_embeddings.assert_called_once_with(
        tenant_id=tenant_id,
        document_id=sample_chunks[0].document_id,
        embeddings=embeddings,
    )
