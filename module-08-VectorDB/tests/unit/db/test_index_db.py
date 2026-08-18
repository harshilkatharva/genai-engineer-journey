from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from semantic_search_eng.db.index_db import IndexDBManager
from semantic_search_eng.models.chunk import Chunk


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.DATABASE_CONNECTION_CONVERSATION_URL = "postgresql://localhost/test"
    return settings


@pytest.fixture
def index_db_manager(monkeypatch, mock_settings):
    monkeypatch.setattr(
        "semantic_search_eng.db.index_db.get_settings",
        lambda: mock_settings,
    )

    mock_tracker = MagicMock()
    monkeypatch.setattr(
        "semantic_search_eng.db.index_db.DBOperationTracker",
        lambda: mock_tracker,
    )

    manager = IndexDBManager()
    return manager, mock_tracker


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            chunk_id="chunk_0",
            document_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
            tenant_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
            document_type="pdf",
            metadata={"page": 1},
            chunk_index=0,
            text="First chunk content.",
            token_count=3,
        ),
        Chunk(
            chunk_id="chunk_1",
            document_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
            tenant_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
            document_type="pdf",
            metadata={"page": 1},
            chunk_index=1,
            text="Second chunk content.",
            token_count=3,
        ),
    ]


@pytest.fixture
def sample_embeddings():
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]


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

    mock_cursor.executemany = AsyncMock()
    mock_conn.commit = AsyncMock()

    return mock_conn, mock_cursor


@pytest.mark.asyncio
async def test_store_index_with_valid_data(
    index_db_manager,
    sample_chunks,
    sample_embeddings,
    mock_db,
):
    manager, tracker = index_db_manager
    mock_conn, mock_cursor = mock_db

    with (
        patch(
            "semantic_search_eng.db.index_db.psycopg.AsyncConnection.connect",
            new=AsyncMock(return_value=mock_conn),
        ),
        patch(
            "semantic_search_eng.db.index_db.register_vector_async",
            new=AsyncMock(),
        ),
    ):
        await manager.store_index(
            tenant_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
            chunks=sample_chunks,
            embeddings=sample_embeddings,
        )

    mock_cursor.executemany.assert_called_once()
    mock_conn.commit.assert_awaited_once()
    tracker.track_index_batch.assert_called_once()

    tracker.track_index_batch.assert_called_once()
    tracker_call = tracker.track_index_batch.call_args.args[0]

    assert tracker_call.tenant_id == "550e8400-e29b-41d4-a716-446655440001"
    assert tracker_call.batch_number == 1
    assert tracker_call.batch_size == 2
    assert tracker_call.total_chunks == 2
    assert tracker_call.insertion_latency_ms >= 0


@pytest.mark.asyncio
async def test_store_index_rejects_mismatched_lengths(
    index_db_manager,
    sample_chunks,
):
    manager, _ = index_db_manager

    with pytest.raises(
        ValueError,
        match="Number of chunks must match number of embeddings",
    ):
        await manager.store_index(
            tenant_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
            chunks=sample_chunks,
            embeddings=[[1.0, 0.0, 0.0]],
        )


@pytest.mark.asyncio
async def test_store_index_processes_batches(
    index_db_manager,
    sample_chunks,
    sample_embeddings,
    mock_db,
):
    manager, tracker = index_db_manager
    mock_conn, mock_cursor = mock_db

    manager.BATCH_SIZE = 1

    many_chunks = sample_chunks * 3
    many_embeddings = sample_embeddings * 3

    with (
        patch(
            "semantic_search_eng.db.index_db.psycopg.AsyncConnection.connect",
            new=AsyncMock(return_value=mock_conn),
        ),
        patch(
            "semantic_search_eng.db.index_db.register_vector_async",
            new=AsyncMock(),
        ),
    ):
        await manager.store_index(
            tenant_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
            chunks=many_chunks,
            embeddings=many_embeddings,
        )

    # 6 chunks / batch size 1 = 6 batches
    assert mock_cursor.executemany.await_count == 6
    assert tracker.track_index_batch.call_count == 6
    mock_conn.commit.assert_awaited_once()

    # Check batch metadata
    tracked_batches = [call.args[0] for call in tracker.track_index_batch.call_args_list]

    assert [batch.batch_number for batch in tracked_batches] == [1, 2, 3, 4, 5, 6]

    assert [batch.batch_size for batch in tracked_batches] == [1, 1, 1, 1, 1, 1]

    assert all(batch.total_chunks == 6 for batch in tracked_batches)


@pytest.mark.asyncio
async def test_store_index_commits_transaction(
    index_db_manager,
    sample_chunks,
    sample_embeddings,
    mock_db,
):
    manager, _ = index_db_manager
    mock_conn, _ = mock_db

    with (
        patch(
            "semantic_search_eng.db.index_db.psycopg.AsyncConnection.connect",
            new=AsyncMock(return_value=mock_conn),
        ),
        patch(
            "semantic_search_eng.db.index_db.register_vector_async",
            new=AsyncMock(),
        ),
    ):
        await manager.store_index(
            tenant_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
            chunks=sample_chunks,
            embeddings=sample_embeddings,
        )

    mock_conn.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_store_index_registers_pgvector(
    index_db_manager,
    sample_chunks,
    sample_embeddings,
    mock_db,
):
    manager, _ = index_db_manager
    mock_conn, _ = mock_db

    with (
        patch(
            "semantic_search_eng.db.index_db.psycopg.AsyncConnection.connect",
            new=AsyncMock(return_value=mock_conn),
        ),
        patch(
            "semantic_search_eng.db.index_db.register_vector_async",
            new=AsyncMock(),
        ) as mock_register_vector,
    ):
        await manager.store_index(
            tenant_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
            chunks=sample_chunks,
            embeddings=sample_embeddings,
        )

    mock_register_vector.assert_awaited_once_with(mock_conn)


@pytest.mark.asyncio
async def test_store_index_inserts_expected_rows(
    index_db_manager,
    sample_chunks,
    sample_embeddings,
    mock_db,
):
    manager, _ = index_db_manager
    _, mock_cursor = mock_db

    tenant_id = UUID("550e8400-e29b-41d4-a716-446655440001")

    with (
        patch(
            "semantic_search_eng.db.index_db.psycopg.AsyncConnection.connect",
            new=AsyncMock(return_value=mock_db[0]),
        ),
        patch(
            "semantic_search_eng.db.index_db.register_vector_async",
            new=AsyncMock(),
        ),
    ):
        await manager.store_index(
            tenant_id=tenant_id,
            chunks=sample_chunks,
            embeddings=sample_embeddings,
        )

    mock_cursor.executemany.assert_awaited_once()

    query, rows = mock_cursor.executemany.await_args.args

    assert "INSERT INTO document_chunks" in query
    assert "tenant_id" in query
    assert "document_id" in query
    assert "chunk_id" in query
    assert "chunk_text" in query
    assert "embedding" in query
    assert "document_type" in query
    assert "metadata" in query

    assert len(rows) == 2

    assert rows[0][0] == tenant_id
    assert rows[0][1] == sample_chunks[0].document_id
    assert rows[0][2] == sample_chunks[0].chunk_id
    assert rows[0][3] == sample_chunks[0].text
    assert rows[0][4] == sample_embeddings[0]
    assert rows[0][5] == sample_chunks[0].document_type

    assert rows[1][0] == tenant_id
    assert rows[1][1] == sample_chunks[1].document_id
    assert rows[1][2] == sample_chunks[1].chunk_id
    assert rows[1][3] == sample_chunks[1].text
    assert rows[1][4] == sample_embeddings[1]
    assert rows[1][5] == sample_chunks[1].document_type
