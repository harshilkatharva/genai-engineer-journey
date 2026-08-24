from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from rag_app.db.retrive_db import RetriveDBManager
from rag_app.models.retrive.retrive_response import RetriveResult


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.DATABASE_CONNECTION_CONVERSATION_URL = "postgresql://localhost/test"
    settings.default_top_k = 5
    return settings


@pytest.fixture
def retrive_db_manager(monkeypatch, mock_settings):
    monkeypatch.setattr(
        "rag_app.db.retrive_db.get_settings",
        lambda: mock_settings,
    )

    mock_tracker = MagicMock()
    monkeypatch.setattr(
        "rag_app.db.retrive_db.DBOperationTracker",
        lambda: mock_tracker,
    )

    manager = RetriveDBManager()
    return manager, mock_tracker


@pytest.fixture
def sample_query_embedding():
    return [1.0, 0.0, 0.0]


@pytest.fixture
def mock_db():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    # async with await psycopg.AsyncConnection.connect(...) as conn
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=None)

    # async with conn.cursor() as cur
    mock_conn.cursor.return_value.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__aexit__ = AsyncMock(return_value=None)

    mock_cursor.execute = AsyncMock()
    mock_cursor.fetchall = AsyncMock()

    return mock_conn, mock_cursor


@pytest.mark.asyncio
async def test_retrive_chunks_returns_results(
    retrive_db_manager,
    sample_query_embedding,
    mock_db,
):
    manager, tracker = retrive_db_manager
    mock_conn, mock_cursor = mock_db

    mock_cursor.fetchall.return_value = [
        (
            "chunk_1",
            "Chunk 1 text",
            0.95,
        ),
        (
            "chunk_2",
            "Chunk 2 text",
            0.87,
        ),
        (
            "chunk_3",
            "Chunk 3 text",
            0.65,
        ),
    ]

    tenant_id = UUID("550e8400-e29b-41d4-a716-446655440001")

    with (
        patch(
            "rag_app.db.retrive_db.psycopg.AsyncConnection.connect",
            new=AsyncMock(return_value=mock_conn),
        ),
        patch(
            "rag_app.db.retrive_db.register_vector_async",
            new=AsyncMock(),
        ),
    ):
        results = await manager.retrive_chunks(
            tenant_id=tenant_id,
            query_embedding=sample_query_embedding,
            top_k=3,
        )

    assert len(results) == 3

    assert isinstance(results[0], RetriveResult)
    assert results[0].chunk_text == "Chunk 1 text"
    assert results[0].similarity_score == 0.95

    assert isinstance(results[2], RetriveResult)
    assert results[2].chunk_text == "Chunk 3 text"
    assert results[2].similarity_score == 0.65

    tracker.track_query.assert_called_once()

    tracker_call = tracker.track_query.call_args.args[0]
    assert tracker_call.tenant_id == str(tenant_id)
    assert tracker_call.top_k == 3
    assert tracker_call.results_count == 3
    assert tracker_call.chunk_ids == ["chunk_1", "chunk_2", "chunk_3"]


@pytest.mark.asyncio
async def test_retrive_chunks_tracks_query(
    retrive_db_manager,
    sample_query_embedding,
    mock_db,
):
    manager, tracker = retrive_db_manager
    mock_conn, mock_cursor = mock_db

    mock_cursor.fetchall.return_value = [
        (
            "chunk_1",
            "Chunk 1",
            0.95,
        ),
        (
            "chunk_2",
            "Chunk 2",
            0.87,
        ),
    ]

    tenant_id = UUID("550e8400-e29b-41d4-a716-446655440001")

    with (
        patch(
            "rag_app.db.retrive_db.psycopg.AsyncConnection.connect",
            new=AsyncMock(return_value=mock_conn),
        ),
        patch(
            "rag_app.db.retrive_db.register_vector_async",
            new=AsyncMock(),
        ),
    ):
        await manager.retrive_chunks(
            tenant_id=tenant_id,
            query_embedding=sample_query_embedding,
            top_k=2,
        )

    tracker.track_query.assert_called_once()

    call_args = tracker.track_query.call_args.args[0]

    assert call_args.tenant_id == str(tenant_id)
    assert call_args.top_k == 2
    assert call_args.results_count == 2
    assert call_args.chunk_ids == ["chunk_1", "chunk_2"]
    assert call_args.query_latency_ms >= 0


@pytest.mark.asyncio
async def test_retrive_chunks_uses_vector_operator(
    retrive_db_manager,
    sample_query_embedding,
    mock_db,
):
    manager, _ = retrive_db_manager
    mock_conn, mock_cursor = mock_db

    mock_cursor.fetchall.return_value = []

    with (
        patch(
            "rag_app.db.retrive_db.psycopg.AsyncConnection.connect",
            new=AsyncMock(return_value=mock_conn),
        ),
        patch(
            "rag_app.db.retrive_db.register_vector_async",
            new=AsyncMock(),
        ),
    ):
        await manager.retrive_chunks(
            tenant_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
            query_embedding=sample_query_embedding,
            top_k=5,
        )

    mock_cursor.execute.assert_awaited_once()

    query, params = mock_cursor.execute.await_args.args

    assert "<=>" in query
    assert "embedding" in query
    assert "ORDER BY embedding <=>" in query
    assert "LIMIT %s" in query
    assert "AND document_type = %s" not in query

    assert params[0] == sample_query_embedding
    assert params[1] == UUID("550e8400-e29b-41d4-a716-446655440001")
    assert params[2] == sample_query_embedding
    assert params[3] == 5


@pytest.mark.asyncio
async def test_retrive_chunks_uses_vector_operator_with_document_type(
    retrive_db_manager,
    sample_query_embedding,
    mock_db,
):
    manager, _ = retrive_db_manager
    mock_conn, mock_cursor = mock_db

    mock_cursor.fetchall.return_value = []

    with (
        patch(
            "rag_app.db.retrive_db.psycopg.AsyncConnection.connect",
            new=AsyncMock(return_value=mock_conn),
        ),
        patch(
            "rag_app.db.retrive_db.register_vector_async",
            new=AsyncMock(),
        ),
    ):
        await manager.retrive_chunks(
            tenant_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
            query_embedding=sample_query_embedding,
            top_k=5,
            document_type="HR Policy",
        )

    mock_cursor.execute.assert_awaited_once()

    query, params = mock_cursor.execute.await_args.args

    assert "<=>" in query
    assert "embedding" in query
    assert "ORDER BY embedding <=>" in query
    assert "LIMIT %s" in query
    assert "AND document_type = %s" in query

    assert params[0] == sample_query_embedding
    assert params[1] == UUID("550e8400-e29b-41d4-a716-446655440001")
    assert params[2] == "HR Policy"
    assert params[3] == sample_query_embedding
    assert params[4] == 5


@pytest.mark.asyncio
async def test_retrive_chunks_with_default_top_k(
    retrive_db_manager,
    sample_query_embedding,
    mock_db,
):
    manager, tracker = retrive_db_manager
    mock_conn, mock_cursor = mock_db

    manager.settings.default_top_k = 10

    mock_cursor.fetchall.return_value = [
        ("chunk_1", "Chunk 1", 0.95),
    ]

    tenant_id = UUID("550e8400-e29b-41d4-a716-446655440001")

    with (
        patch(
            "rag_app.db.retrive_db.psycopg.AsyncConnection.connect",
            new=AsyncMock(return_value=mock_conn),
        ),
        patch(
            "rag_app.db.retrive_db.register_vector_async",
            new=AsyncMock(),
        ),
    ):
        results = await manager.retrive_chunks(
            tenant_id=tenant_id,
            query_embedding=sample_query_embedding,
            top_k=None,
        )

    assert len(results) == 1

    # The SQL query receives None when top_k=None.
    # This is distinct from the value recorded in DBQueryTracker.
    _, params = mock_cursor.execute.await_args.args
    assert params[3] is None

    tracker_call = tracker.track_query.call_args.args[0]
    assert tracker_call.top_k == 10


@pytest.mark.asyncio
async def test_retrive_chunks_registers_pgvector(
    retrive_db_manager,
    sample_query_embedding,
    mock_db,
):
    manager, _ = retrive_db_manager
    mock_conn, _ = mock_db

    with (
        patch(
            "rag_app.db.retrive_db.psycopg.AsyncConnection.connect",
            new=AsyncMock(return_value=mock_conn),
        ),
        patch(
            "rag_app.db.retrive_db.register_vector_async",
            new=AsyncMock(),
        ) as mock_register_vector,
    ):
        await manager.retrive_chunks(
            tenant_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
            query_embedding=sample_query_embedding,
            top_k=5,
        )

    mock_register_vector.assert_awaited_once_with(mock_conn)


@pytest.mark.asyncio
async def test_retrive_chunks_handles_empty_results(
    retrive_db_manager,
    sample_query_embedding,
    mock_db,
):
    manager, tracker = retrive_db_manager
    mock_conn, mock_cursor = mock_db

    mock_cursor.fetchall.return_value = []

    with (
        patch(
            "rag_app.db.retrive_db.psycopg.AsyncConnection.connect",
            new=AsyncMock(return_value=mock_conn),
        ),
        patch(
            "rag_app.db.retrive_db.register_vector_async",
            new=AsyncMock(),
        ),
    ):
        results = await manager.retrive_chunks(
            tenant_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
            query_embedding=sample_query_embedding,
            top_k=5,
        )

    assert results == []

    tracker_call = tracker.track_query.call_args.args[0]

    assert tracker_call.results_count == 0
    assert tracker_call.chunk_ids == []
    assert tracker_call.top_k == 5


@pytest.mark.asyncio
async def test_retrive_chunks_executes_query_with_expected_parameters(
    retrive_db_manager,
    sample_query_embedding,
    mock_db,
):
    manager, _ = retrive_db_manager
    mock_conn, mock_cursor = mock_db

    mock_cursor.fetchall.return_value = []

    tenant_id = UUID("550e8400-e29b-41d4-a716-446655440001")

    with (
        patch(
            "rag_app.db.retrive_db.psycopg.AsyncConnection.connect",
            new=AsyncMock(return_value=mock_conn),
        ),
        patch(
            "rag_app.db.retrive_db.register_vector_async",
            new=AsyncMock(),
        ),
    ):
        await manager.retrive_chunks(
            tenant_id=tenant_id,
            query_embedding=sample_query_embedding,
            top_k=7,
        )

    mock_cursor.execute.assert_awaited_once()

    query, params = mock_cursor.execute.await_args.args

    assert "FROM document_chunks" in query
    assert "WHERE tenant_id = %s" in query
    assert "LIMIT %s" in query

    assert params == (
        sample_query_embedding,
        tenant_id,
        sample_query_embedding,
        7,
    )
