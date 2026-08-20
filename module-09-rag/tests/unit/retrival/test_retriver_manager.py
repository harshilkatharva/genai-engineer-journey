"""Updated tests for RetriverManager with async database integration"""

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from pgvector import Vector

from rag_app.models.retrive.retrive_response import RetriveResult
from rag_app.retrival.retriver_manager import RetriverManager


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.embedding_model = "all-MiniLM-L6-v2"
    settings.embedding_cost_per_million_tokens = 0.0
    settings.default_top_k = 5
    settings.max_top_k = 100
    return settings


@pytest.fixture
def retriver_manager(monkeypatch, mock_settings):
    monkeypatch.setattr(
        "rag_app.retrival.retriver_manager.get_settings",
        lambda: mock_settings,
    )

    mock_embedding_manager = MagicMock()
    mock_embedding_manager.embed_query = MagicMock(return_value=[1.0, 0.0, 0.0])

    monkeypatch.setattr(
        "rag_app.retrival.retriver_manager.EmbeddingManager",
        lambda: mock_embedding_manager,
    )

    mock_retrive_db_manager = AsyncMock()
    monkeypatch.setattr(
        "rag_app.retrival.retriver_manager.RetriveDBManager",
        lambda: mock_retrive_db_manager,
    )

    mock_tracker_logger = MagicMock()
    monkeypatch.setattr(
        "rag_app.retrival.retriver_manager.RetriveTrackerLogger",
        lambda: mock_tracker_logger,
    )

    manager = RetriverManager()
    return manager, mock_embedding_manager, mock_retrive_db_manager, mock_tracker_logger


@pytest.fixture
def sample_results():
    return [
        RetriveResult(chunk_text="First result", similarity_score=0.95),
        RetriveResult(chunk_text="Second result", similarity_score=0.87),
        RetriveResult(chunk_text="Third result", similarity_score=0.65),
    ]


@pytest.mark.asyncio
async def test_retrieve_embeds_query(retriver_manager):
    manager, embedding_manager, retrive_db_manager, _ = retriver_manager

    retrive_db_manager.retrive_chunks.return_value = []

    await manager.retrieve(
        tenant_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
        query="test query",
        top_k=5,
    )

    embedding_manager.embed_query.assert_called_once_with("test query")


@pytest.mark.asyncio
async def test_retrieve_queries_database(retriver_manager, sample_results):
    manager, embedding_manager, retrive_db_manager, _ = retriver_manager

    embedding_manager.embed_query.return_value = [1.0, 0.0, 0.0]
    retrive_db_manager.retrive_chunks.return_value = sample_results

    tenant_id = UUID("550e8400-e29b-41d4-a716-446655440001")

    await manager.retrieve(
        tenant_id=tenant_id,
        query="test query",
        top_k=3,
    )

    retrive_db_manager.retrive_chunks.assert_called_once()
    call_args = retrive_db_manager.retrive_chunks.call_args.args

    assert call_args[0] == tenant_id
    assert call_args[1] == Vector([1.0, 0.0, 0.0])
    assert call_args[2] == 3


@pytest.mark.asyncio
async def test_retrieve_returns_database_results(retriver_manager, sample_results):
    manager, embedding_manager, retrive_db_manager, _ = retriver_manager

    embedding_manager.embed_query.return_value = [1.0, 0.0, 0.0]
    retrive_db_manager.retrive_chunks.return_value = sample_results

    results = await manager.retrieve(
        tenant_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
        query="test query",
        top_k=3,
    )

    assert len(results) == 3
    assert results[0].chunk_text == "First result"
    assert results[0].similarity_score == 0.95


@pytest.mark.asyncio
async def test_retrieve_tracks_metrics(retriver_manager, sample_results):
    manager, embedding_manager, retrive_db_manager, tracker_logger = retriver_manager

    embedding_manager.embed_query.return_value = [1.0, 0.0, 0.0]
    retrive_db_manager.retrive_chunks.return_value = sample_results

    await manager.retrieve(
        tenant_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
        query="test query",
        top_k=3,
    )

    assert tracker_logger.track.called
    call_args = tracker_logger.track.call_args[0][0]
    assert call_args.tenant_id == "550e8400-e29b-41d4-a716-446655440001"
    assert call_args.query == "test query"
    assert call_args.top_k == 3
    assert call_args.results_count == 3


@pytest.mark.asyncio
async def test_retrieve_tracks_metrics_with_document_types(retriver_manager, sample_results):
    manager, embedding_manager, retrive_db_manager, tracker_logger = retriver_manager

    embedding_manager.embed_query.return_value = [1.0, 0.0, 0.0]
    retrive_db_manager.retrive_chunks.return_value = sample_results

    await manager.retrieve(
        tenant_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
        query="test query",
        top_k=3,
        document_type="HR Policy",
    )

    assert tracker_logger.track.called
    call_args = tracker_logger.track.call_args[0][0]
    assert call_args.tenant_id == "550e8400-e29b-41d4-a716-446655440001"
    assert call_args.query == "test query"
    assert call_args.top_k == 3
    assert call_args.results_count == 3


@pytest.mark.asyncio
async def test_retrieve_rejects_empty_query(retriver_manager):
    manager, embedding_manager, _, _ = retriver_manager

    with pytest.raises(ValueError, match="Query cannot be empty"):
        await manager.retrieve(
            tenant_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
            query="   ",
            top_k=5,
        )

    embedding_manager.embed_query.assert_not_called()


@pytest.mark.asyncio
async def test_retrieve_rejects_top_k_exceeding_maximum(retriver_manager, mock_settings):
    manager, embedding_manager, _, _ = retriver_manager
    mock_settings.max_top_k = 10

    with pytest.raises(ValueError, match="Can not retrive more than"):
        await manager.retrieve(
            tenant_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
            query="test query",
            top_k=15,
        )

    embedding_manager.embed_query.assert_not_called()


@pytest.mark.asyncio
async def test_retrieve_with_default_top_k(retriver_manager, sample_results, mock_settings):
    manager, embedding_manager, retrive_db_manager, _ = retriver_manager
    mock_settings.default_top_k = 10

    embedding_manager.embed_query.return_value = [1.0, 0.0, 0.0]
    retrive_db_manager.retrive_chunks.return_value = sample_results

    await manager.retrieve(
        tenant_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
        query="test query",
        top_k=None,
    )

    # Check that default top_k is used when None is passed
    call_args = retrive_db_manager.retrive_chunks.call_args.args

    assert call_args[1] == Vector([1.0, 0.0, 0.0])
    assert call_args[2] == 10


@pytest.mark.asyncio
async def test_retrieve_handles_empty_results(retriver_manager):
    manager, embedding_manager, retrive_db_manager, tracker_logger = retriver_manager

    embedding_manager.embed_query.return_value = [1.0, 0.0, 0.0]
    retrive_db_manager.retrive_chunks.return_value = []

    results = await manager.retrieve(
        tenant_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
        query="query with no results",
        top_k=5,
    )

    assert results == []
    assert tracker_logger.track.called
    call_args = tracker_logger.track.call_args[0][0]
    assert call_args.results_count == 0
