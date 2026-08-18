from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from semantic_search_eng.db.upsert_db import UpsertDBManager


@pytest.fixture
def db_manager():
    manager = UpsertDBManager()

    manager.settings = MagicMock()
    manager.settings.DATABASE_CONNECTION_CONVERSATION_URL = "postgresql://test:test@localhost/test"

    manager.db_tracker = MagicMock()

    return manager


@pytest.fixture
def mock_db():
    conn = MagicMock()
    cursor = MagicMock()

    # Connection is an async context manager.
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=None)

    # cursor() itself is NOT awaited.
    # It returns an async context manager.
    conn.cursor.return_value = cursor

    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=None)

    # These operations ARE awaited.
    cursor.execute = AsyncMock()
    conn.commit = AsyncMock()

    return conn, cursor


@pytest.mark.asyncio
async def test_upsert_chunks_success(db_manager, mock_db):
    conn, cursor = mock_db

    tenant_id = uuid4()

    document_id_1 = uuid4()
    document_id_2 = uuid4()

    updated_chunks = [
        "updated text 1",
        "updated text 2",
    ]

    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    document_ids = [
        document_id_1,
        document_id_2,
    ]

    chunk_ids = [
        "chunk_1",
        "chunk_2",
    ]

    # Both UPDATEs succeed.
    cursor.rowcount = 1

    with (
        patch(
            "semantic_search_eng.db.upsert_db.psycopg.AsyncConnection.connect",
            new_callable=AsyncMock,
            return_value=conn,
        ) as mock_connect,
        patch(
            "semantic_search_eng.db.upsert_db.register_vector_async",
            new_callable=AsyncMock,
        ) as mock_register_vector,
    ):
        result = await db_manager.upsert_chunks(
            tenant_id=tenant_id,
            updated_chunks=updated_chunks,
            updated_embeddings=embeddings,
            document_ids=document_ids,
            chunk_ids=chunk_ids,
        )

    assert result == ["chunk_1", "chunk_2"]

    mock_connect.assert_awaited_once_with("postgresql://test:test@localhost/test")

    mock_register_vector.assert_awaited_once_with(conn)

    assert cursor.execute.await_count == 2

    conn.commit.assert_awaited_once()

    db_manager.db_tracker.track_query.assert_called_once()
