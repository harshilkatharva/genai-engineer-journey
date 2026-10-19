from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genai_core.exceptions import (
    IncompatibleEmbeddingError,
    MissingExtraDependencyError,
    VectorStoreError,
)
from genai_core.models import DocumentChunk, RetrievalQuery, RetrievalResult
from genai_core.resilience import is_retryable_error
from genai_core.retrieval import (
    PGVectorStoreAdapter,
    VectorStoreAdapter,
    check_postgres_dependencies,
)
from genai_core.services import RetrievalService


def _create_mock_connection(rows: list[tuple] | None = None, rowcount: int = 1) -> MagicMock:
    """Helper to create a mock async connection with cursor context manager."""
    conn = MagicMock()
    cur = AsyncMock()
    cur.fetchall = AsyncMock(return_value=rows or [])
    cur.execute = AsyncMock()
    cur.executemany = AsyncMock()
    cur.rowcount = rowcount

    # Setup async context manager for cursor: async with conn.cursor() as cur:
    cursor_cm = MagicMock()
    cursor_cm.__aenter__ = AsyncMock(return_value=cur)
    cursor_cm.__aexit__ = AsyncMock(return_value=None)
    conn.cursor = MagicMock(return_value=cursor_cm)
    conn.commit = AsyncMock()

    return conn


def test_pgvector_adapter_protocol_conformance() -> None:
    conn = _create_mock_connection()
    adapter = PGVectorStoreAdapter(dimensions=3, connection=conn)
    assert isinstance(adapter, VectorStoreAdapter)
    assert adapter.dimensions == 3


def test_pgvector_adapter_invalid_dimension() -> None:
    conn = _create_mock_connection()
    with pytest.raises(ValueError, match="dimensions must be greater than 0"):
        PGVectorStoreAdapter(dimensions=0, connection=conn)


def test_pgvector_adapter_sql_identifier_safety() -> None:
    conn = _create_mock_connection()

    # Valid identifier
    adapter = PGVectorStoreAdapter(
        dimensions=3,
        connection=conn,
        table_name="my_chunks_v2",
        embedding_column="vector_col",
        id_column="doc_id",
        text_column="raw_text",
        metadata_column="meta_json",
        tenant_id_column="tenant_uuid",
        conflict_target="doc_id",
    )
    assert adapter.table_name == "my_chunks_v2"

    # SQL injection attempts in identifiers
    with pytest.raises(ValueError, match="Invalid SQL identifier for table_name"):
        PGVectorStoreAdapter(
            dimensions=3,
            connection=conn,
            table_name="document_chunks; DROP TABLE users;--",
        )

    with pytest.raises(ValueError, match="Invalid SQL identifier for embedding_column"):
        PGVectorStoreAdapter(
            dimensions=3,
            connection=conn,
            embedding_column="col name with spaces",
        )

    with pytest.raises(ValueError, match="Invalid SQL identifier for id_column"):
        PGVectorStoreAdapter(
            dimensions=3,
            connection=conn,
            id_column="123_invalid_start",
        )


def test_pgvector_adapter_missing_dependencies() -> None:
    # Test check_postgres_dependencies raises MissingExtraDependencyError when imports fail
    with (
        patch.dict("sys.modules", {"psycopg": None, "pgvector": None}),
        pytest.raises(MissingExtraDependencyError) as exc_info,
    ):
        check_postgres_dependencies()
    err_msg = str(exc_info.value)
    assert "postgres" in err_msg
    assert "psycopg" in err_msg

    # Test initialization with check_dependencies=True raises MissingExtraDependencyError
    with (
        patch.dict("sys.modules", {"psycopg": None, "pgvector": None}),
        pytest.raises(MissingExtraDependencyError),
    ):
        PGVectorStoreAdapter(dimensions=3, check_dependencies=True)


def test_pgvector_adapter_missing_connection_and_pool() -> None:
    # If no connection or pool is passed, it checks dependencies and raises ValueError
    with (
        patch("genai_core.retrieval.pgvector_store.check_postgres_dependencies"),
        pytest.raises(ValueError, match="PGVectorStoreAdapter requires an injected"),
    ):
        PGVectorStoreAdapter(dimensions=3)


@pytest.mark.asyncio
async def test_pgvector_adapter_search_result_mapping() -> None:
    mock_rows = [
        ("c1", "First match chunk", json.dumps({"category": "ai", "page": 1}), 0.95),
        ("c2", "Second match chunk", {"category": "ai", "page": 2}, 0.82),  # Already dict
    ]
    conn = _create_mock_connection(rows=mock_rows)
    adapter = PGVectorStoreAdapter(dimensions=3, connection=conn)

    results = await adapter.search(query_vector=[0.1, 0.2, 0.3], top_k=5)

    assert len(results) == 2
    r1, r2 = results[0], results[1]

    assert isinstance(r1, RetrievalResult)
    assert r1.id == "c1"
    assert r1.content == "First match chunk"
    assert r1.metadata == {"category": "ai", "page": 1}
    assert r1.score == 0.95
    assert r1.rank == 1

    assert r2.id == "c2"
    assert r2.content == "Second match chunk"
    assert r2.metadata == {"category": "ai", "page": 2}
    assert r2.score == 0.82
    assert r2.rank == 2


@pytest.mark.asyncio
async def test_pgvector_adapter_search_dimension_validation() -> None:
    conn = _create_mock_connection()
    adapter = PGVectorStoreAdapter(dimensions=3, connection=conn)

    with pytest.raises(IncompatibleEmbeddingError) as exc_info:
        await adapter.search(query_vector=[0.1, 0.2])  # 2 dims vs 3 expected
    assert exc_info.value.requested == 2
    assert exc_info.value.supported == 3


@pytest.mark.asyncio
async def test_pgvector_adapter_search_top_k_zero() -> None:
    conn = _create_mock_connection()
    adapter = PGVectorStoreAdapter(dimensions=3, connection=conn)

    results = await adapter.search(query_vector=[0.1, 0.2, 0.3], top_k=0)
    assert results == []


@pytest.mark.asyncio
async def test_pgvector_adapter_sql_parameterization_and_filtering() -> None:
    conn = _create_mock_connection(rows=[])
    adapter = PGVectorStoreAdapter(
        dimensions=3,
        connection=conn,
        table_name="test_chunks",
        tenant_id_column="tenant_id",
    )

    query_vec = [1.0, 0.0, 0.0]
    filters = {"tenant_id": "tenant-abc-123", "dept": "finance"}
    score_threshold = 0.75
    top_k = 10

    await adapter.search(
        query_vector=query_vec,
        top_k=top_k,
        score_threshold=score_threshold,
        filters=filters,
    )

    # Verify query execution on cursor
    cur = await conn.cursor().__aenter__()
    assert cur.execute.called
    call_args = cur.execute.call_args
    sql_text, params = call_args[0]

    # Verify SQL text structure and safety
    assert "FROM test_chunks" in sql_text
    assert "tenant_id = %s" in sql_text
    assert "metadata @> %s::jsonb" in sql_text
    assert "1 - (embedding <=> %s) >= %s" in sql_text
    assert "ORDER BY embedding <=> %s ASC" in sql_text
    assert "LIMIT %s" in sql_text

    # Verify parameters are passed safely as tuple
    assert params[0] == query_vec
    assert params[1] == "tenant-abc-123"
    assert params[2] == json.dumps({"dept": "finance"})
    assert params[3] == query_vec
    assert params[4] == score_threshold
    assert params[5] == query_vec
    assert params[6] == top_k


@pytest.mark.asyncio
async def test_pgvector_adapter_upsert_validation_and_execution() -> None:
    conn = _create_mock_connection()
    adapter = PGVectorStoreAdapter(dimensions=3, connection=conn)

    # Empty list returns 0
    assert await adapter.upsert([]) == 0

    # Missing embedding vector raises ValueError
    chunk_no_emb = DocumentChunk(id="c0", content="Text without vector", embedding=None)
    with pytest.raises(ValueError, match="missing embedding vector"):
        await adapter.upsert([chunk_no_emb])

    # Dimension mismatch raises IncompatibleEmbeddingError
    chunk_bad_dim = DocumentChunk(id="c1", content="Bad dimension", embedding=[1.0, 2.0])
    with pytest.raises(IncompatibleEmbeddingError) as exc_info:
        await adapter.upsert([chunk_bad_dim])
    assert exc_info.value.requested == 2
    assert exc_info.value.supported == 3

    # Valid chunks upsert
    valid_chunks = [
        DocumentChunk(id="c1", content="Text 1", embedding=[1.0, 0.0, 0.0], metadata={"a": 1}),
        DocumentChunk(id="c2", content="Text 2", embedding=[0.0, 1.0, 0.0], metadata={"b": 2}),
    ]
    count = await adapter.upsert(valid_chunks)
    assert count == 2

    cur = await conn.cursor().__aenter__()
    assert cur.executemany.called
    sql_text, rows = cur.executemany.call_args[0]
    assert "INSERT INTO document_chunks" in sql_text
    assert "ON CONFLICT (chunk_id)" in sql_text
    assert "DO UPDATE SET" in sql_text
    assert len(rows) == 2
    assert rows[0][0] == "c1"
    assert rows[0][2] == [1.0, 0.0, 0.0]
    assert conn.commit.called


@pytest.mark.asyncio
async def test_pgvector_adapter_delete_execution() -> None:
    conn = _create_mock_connection(rowcount=3)
    adapter = PGVectorStoreAdapter(dimensions=3, connection=conn)

    # Empty delete returns 0
    assert await adapter.delete([]) == 0

    deleted = await adapter.delete(["c1", "c2", "c3"])
    assert deleted == 3

    cur = await conn.cursor().__aenter__()
    assert cur.execute.called
    sql_text, params = cur.execute.call_args[0]
    assert "DELETE FROM document_chunks WHERE chunk_id = ANY(%s)" in sql_text
    assert params == (["c1", "c2", "c3"],)
    assert conn.commit.called


@pytest.mark.asyncio
async def test_pgvector_adapter_transient_failure_and_resilience() -> None:
    # Simulate connection reset error
    transient_conn = MagicMock()
    cur = AsyncMock()
    cur.execute.side_effect = ConnectionResetError("Connection reset by peer")
    cursor_cm = MagicMock()
    cursor_cm.__aenter__ = AsyncMock(return_value=cur)
    cursor_cm.__aexit__ = AsyncMock(return_value=None)
    transient_conn.cursor = MagicMock(return_value=cursor_cm)

    adapter = PGVectorStoreAdapter(dimensions=3, connection=transient_conn)

    with pytest.raises(VectorStoreError) as exc_info:
        await adapter.search(query_vector=[1.0, 0.0, 0.0])

    err = exc_info.value
    assert err.backend is not None and "pgvector" in err.backend
    assert err.retryable is True
    assert is_retryable_error(err) is True


@pytest.mark.asyncio
async def test_pgvector_adapter_permanent_failure() -> None:
    # Simulate table relation does not exist error
    perm_conn = MagicMock()
    cur = AsyncMock()
    cur.execute.side_effect = Exception('relation "document_chunks" does not exist')
    cursor_cm = MagicMock()
    cursor_cm.__aenter__ = AsyncMock(return_value=cur)
    cursor_cm.__aexit__ = AsyncMock(return_value=None)
    perm_conn.cursor = MagicMock(return_value=cursor_cm)

    adapter = PGVectorStoreAdapter(dimensions=3, connection=perm_conn)

    with pytest.raises(VectorStoreError) as exc_info:
        await adapter.search(query_vector=[1.0, 0.0, 0.0])

    err = exc_info.value
    assert err.retryable is False
    assert is_retryable_error(err) is False


@pytest.mark.asyncio
async def test_pgvector_adapter_integration_with_retrieval_service() -> None:
    mock_rows = [
        ("chunk_1", "Result text from postgres", json.dumps({"source": "manual"}), 0.91),
    ]
    conn = _create_mock_connection(rows=mock_rows)
    pg_store = PGVectorStoreAdapter(dimensions=3, connection=conn)

    fake_embedding_service = MagicMock()
    fake_embedding_service.dimensions = 3
    fake_embedding_service.embed_query = AsyncMock(return_value=[1.0, 0.0, 0.0])

    retrieval_service = RetrievalService(store=pg_store, embedding_service=fake_embedding_service)

    response = await retrieval_service.search(RetrievalQuery(query="test database search", top_k=3))
    assert response.total_results == 1
    assert response.results[0].id == "chunk_1"
    assert response.results[0].content == "Result text from postgres"
    assert response.results[0].score == 0.91
    assert response.results[0].rank == 1
