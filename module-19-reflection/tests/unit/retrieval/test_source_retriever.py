"""Unit tests for SourceRetriever."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from genai_core import DocumentChunk, RetrievalResponse, RetrievalResult

from self_improving_report_writer.core.exceptions import RetrievalError
from self_improving_report_writer.retrieval.source_retriever import SourceRetriever


@pytest.fixture
def mock_retrieval_service() -> AsyncMock:
    """Mock genai-core RetrievalService."""
    service = AsyncMock()
    service.search = AsyncMock()
    service.index_documents = AsyncMock(return_value=2)
    return service


async def test_retrieve_sources_success(mock_retrieval_service: AsyncMock) -> None:
    """Retrieve sources should map RetrievalResult objects into SourceReferences."""
    mock_retrieval_service.search.return_value = RetrievalResponse(
        results=[
            RetrievalResult(
                id="chunk_1",
                content="Quantum speedup content",
                score=0.92,
                rank=1,
                source_id="src_1",
                metadata={"title": "Quantum Paper"},
            ),
            RetrievalResult(
                id="chunk_2",
                content="Classical baseline content",
                score=0.85,
                rank=2,
                source_id="src_2",
                metadata={"title": "Classical Paper"},
            ),
        ],
        total_results=2,
        latency_ms=12.5,
    )

    retriever = SourceRetriever(retrieval_service=mock_retrieval_service)
    sources = await retriever.retrieve_sources(topic="Quantum Optimization", top_k=5)

    assert len(sources) == 2
    assert sources[0].source_id == "src_1"
    assert sources[0].chunk_id == "chunk_1"
    assert sources[0].title == "Quantum Paper"
    assert sources[0].content == "Quantum speedup content"
    assert sources[0].score == 0.92

    assert sources[1].source_id == "src_2"
    assert sources[1].chunk_id == "chunk_2"
    assert sources[1].score == 0.85


async def test_retrieve_sources_min_score_filter(mock_retrieval_service: AsyncMock) -> None:
    """Chunks below min_score should be filtered out."""
    mock_retrieval_service.search.return_value = RetrievalResponse(
        results=[
            RetrievalResult(
                id="c1",
                content="High score",
                score=0.85,
                rank=1,
                source_id="s1",
            ),
            RetrievalResult(
                id="c2",
                content="Low score",
                score=0.35,
                rank=2,
                source_id="s2",
            ),
        ],
        total_results=2,
        latency_ms=10.0,
    )

    retriever = SourceRetriever(retrieval_service=mock_retrieval_service)
    sources = await retriever.retrieve_sources(topic="Topic", min_score=0.50)

    assert len(sources) == 1
    assert sources[0].chunk_id == "c1"


async def test_retrieve_sources_empty_topic_raises() -> None:
    """Empty or whitespace-only topic should raise RetrievalError."""
    retriever = SourceRetriever()
    with pytest.raises(RetrievalError, match="Topic cannot be empty"):
        await retriever.retrieve_sources(topic="   ")


async def test_retrieve_sources_service_error_handling(
    mock_retrieval_service: AsyncMock,
) -> None:
    """Underlying service exceptions should be wrapped in RetrievalError."""
    mock_retrieval_service.search.side_effect = RuntimeError("Database timeout")

    retriever = SourceRetriever(retrieval_service=mock_retrieval_service)
    with pytest.raises(RetrievalError, match="Evidence retrieval failed"):
        await retriever.retrieve_sources(topic="Faulty Query")


async def test_index_sources_convenience(mock_retrieval_service: AsyncMock) -> None:
    """Index sources should convert inputs and invoke index_documents."""
    retriever = SourceRetriever(retrieval_service=mock_retrieval_service)
    docs = [
        DocumentChunk(id="c1", content="Chunk 1", source_id="s1"),
        {"chunk_id": "c2", "content": "Chunk 2", "source_id": "s2"},
    ]
    count = await retriever.index_sources(docs)
    assert count == 2
    mock_retrieval_service.index_documents.assert_called_once()


def test_default_retriever_initializes_with_pgvector_and_embedding_service() -> None:
    """Default SourceRetriever should configure PGVectorStoreAdapter and EmbeddingService."""
    from genai_core import PGVectorStoreAdapter

    retriever = SourceRetriever()
    assert retriever.config.vector_store_backend == "pgvector"
    assert retriever.config.database_url == "postgresql://localhost:5432/ai_search"
    assert isinstance(retriever.store, PGVectorStoreAdapter)
    assert retriever.store.dimensions == 384
    assert retriever.embedding_service is not None
    assert retriever.embedding_service.dimensions == 384


def test_retriever_with_memory_backend_initializes_in_memory_store() -> None:
    """ReportWriterConfig with memory backend should initialize InMemoryVectorStore."""
    from genai_core import InMemoryVectorStore

    from self_improving_report_writer.core.config import ReportWriterConfig

    config = ReportWriterConfig(vector_store_backend="memory")
    retriever = SourceRetriever(config=config)
    assert isinstance(retriever.store, InMemoryVectorStore)
    assert retriever.store.dimensions == 384
    assert retriever.embedding_service is not None


def test_retriever_with_custom_injected_vector_store() -> None:
    """Explicitly injected vector store should take precedence over backend config."""
    from genai_core import InMemoryVectorStore

    custom_store = InMemoryVectorStore(dimensions=384)
    retriever = SourceRetriever(vector_store=custom_store)
    assert retriever.store is custom_store


async def test_retriever_offline_indexing_and_search() -> None:
    """Retriever should index and retrieve chunks offline with in-memory store and mock embeddings."""
    from unittest.mock import MagicMock

    from genai_core import InMemoryVectorStore

    mock_embedding = MagicMock()
    mock_embedding.dimensions = 384
    mock_embedding.embed_query = AsyncMock(return_value=[0.1] * 384)
    mock_embedding.embed_documents = AsyncMock(return_value=[[0.1] * 384])

    store = InMemoryVectorStore(dimensions=384)
    retriever = SourceRetriever(vector_store=store, embedding_service=mock_embedding)

    indexed = await retriever.index_sources(
        [{"chunk_id": "c1", "content": "Offline test content", "source_id": "src_offline"}]
    )
    assert indexed == 1

    sources = await retriever.retrieve_sources(topic="test content", top_k=1)
    assert len(sources) == 1
    assert sources[0].chunk_id == "c1"
    assert sources[0].source_id == "src_offline"
    assert sources[0].content == "Offline test content"


async def test_index_sources_error_wrapped_in_retrieval_error(
    mock_retrieval_service: AsyncMock,
) -> None:
    """Underlying indexing exceptions should be caught and re-raised as RetrievalError."""
    mock_retrieval_service.index_documents.side_effect = RuntimeError("Database write failure")

    retriever = SourceRetriever(retrieval_service=mock_retrieval_service)
    with pytest.raises(RetrievalError, match="Evidence indexing failed"):
        await retriever.index_sources([{"chunk_id": "c1", "content": "content"}])


async def test_missing_embedding_service_fails_dense_search_non_silently() -> None:
    """Dense retrieval without an EmbeddingService must raise RetrievalError, not fall back silently."""
    from genai_core import InMemoryVectorStore
    from genai_core import RetrievalService as CoreRetrievalService

    store = InMemoryVectorStore(dimensions=384)
    # Service without embedding service
    service = CoreRetrievalService(store=store, embedding_service=None)
    retriever = SourceRetriever(retrieval_service=service)

    with pytest.raises(RetrievalError, match="Evidence retrieval failed"):
        await retriever.retrieve_sources("Requires dense embedding")


async def test_missing_embedding_service_fails_indexing_non_silently() -> None:
    """Indexing unembedded chunks without an EmbeddingService must raise RetrievalError non-silently."""
    from genai_core import InMemoryVectorStore
    from genai_core import RetrievalService as CoreRetrievalService

    store = InMemoryVectorStore(dimensions=384)
    service = CoreRetrievalService(store=store, embedding_service=None)
    retriever = SourceRetriever(retrieval_service=service)

    with pytest.raises(RetrievalError, match="Evidence indexing failed"):
        await retriever.index_sources([{"chunk_id": "c1", "content": "Unembedded text"}])


async def test_pgvector_connection_factory_uses_autocommit_and_closes_on_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PGVector read queries must connect with autocommit=True and close cleanly (FINDING-12 Option A)."""
    from unittest.mock import MagicMock

    import psycopg
    from genai_core import PGVectorStoreAdapter

    mock_cursor = MagicMock()
    mock_cursor.execute = AsyncMock()
    mock_cursor.fetchall = AsyncMock(return_value=[("c1", "Chunk text", {"source_id": "s1"}, 0.95)])

    cursor_cm = MagicMock()
    cursor_cm.__aenter__ = AsyncMock(return_value=mock_cursor)
    cursor_cm.__aexit__ = AsyncMock(return_value=None)

    mock_conn = MagicMock()
    mock_conn.autocommit = True
    mock_conn.cursor = MagicMock(return_value=cursor_cm)
    mock_conn.rollback = AsyncMock()
    mock_conn.close = AsyncMock()

    connect_mock = AsyncMock(return_value=mock_conn)
    monkeypatch.setattr(psycopg.AsyncConnection, "connect", connect_mock)

    retriever = SourceRetriever()
    assert isinstance(retriever.store, PGVectorStoreAdapter)

    results = await retriever.store.search([0.1] * 384, top_k=2)

    assert len(results) == 1
    assert results[0].id == "c1"
    connect_mock.assert_awaited_once_with(retriever.config.database_url, autocommit=True)
    mock_conn.rollback.assert_not_awaited()
    mock_conn.close.assert_awaited_once()


async def test_pgvector_connection_factory_rolls_back_and_closes_on_query_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When autocommit is False or query fails, PG connection must rollback and close cleanly."""
    from unittest.mock import MagicMock

    import psycopg
    from genai_core import PGVectorStoreAdapter, VectorStoreError

    mock_cursor = MagicMock()
    mock_cursor.execute = AsyncMock(side_effect=RuntimeError("Query aborted"))

    cursor_cm = MagicMock()
    cursor_cm.__aenter__ = AsyncMock(return_value=mock_cursor)
    cursor_cm.__aexit__ = AsyncMock(return_value=None)

    mock_conn = MagicMock()
    mock_conn.autocommit = False
    mock_conn.cursor = MagicMock(return_value=cursor_cm)
    mock_conn.rollback = AsyncMock()
    mock_conn.close = AsyncMock()

    connect_mock = AsyncMock(return_value=mock_conn)
    monkeypatch.setattr(psycopg.AsyncConnection, "connect", connect_mock)

    retriever = SourceRetriever()
    assert isinstance(retriever.store, PGVectorStoreAdapter)

    with pytest.raises(VectorStoreError, match="Query aborted"):
        await retriever.store.search([0.1] * 384, top_k=2)

    connect_mock.assert_awaited_once_with(retriever.config.database_url, autocommit=True)
    mock_conn.rollback.assert_awaited_once()
    mock_conn.close.assert_awaited_once()
