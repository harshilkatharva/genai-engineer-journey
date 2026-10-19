from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genai_core.exceptions import (
    IncompatibleEmbeddingError,
    RetrievalError,
    RetrievalTimeoutError,
    UnsupportedSearchModeError,
)
from genai_core.models import (
    DocumentChunk,
    RetrievalQuery,
    RetrievalResponse,
)
from genai_core.observability import ObservabilityEvent
from genai_core.retrieval import InMemoryVectorStore
from genai_core.services import RetrievalService


@pytest.fixture
def fake_embedding_service() -> MagicMock:
    emb_svc = MagicMock()
    emb_svc.dimensions = 3
    # Dummy deterministic embeddings
    emb_svc.embed_query = AsyncMock(return_value=[1.0, 0.0, 0.0])
    emb_svc.embed_documents = AsyncMock(return_value=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return emb_svc


def test_retrieval_service_initialization_dimension_mismatch(
    fake_embedding_service: MagicMock,
) -> None:
    store = InMemoryVectorStore(dimensions=4)
    with pytest.raises(IncompatibleEmbeddingError) as exc_info:
        RetrievalService(store=store, embedding_service=fake_embedding_service)
    assert exc_info.value.requested == 3
    assert exc_info.value.supported == 4


def test_retrieval_service_initialization_without_embedding_service() -> None:
    store = InMemoryVectorStore(dimensions=3)
    service = RetrievalService(store=store)
    assert service.embedding_service is None
    assert service.store.dimensions == 3


@pytest.mark.asyncio
async def test_retrieval_service_text_query_flow(fake_embedding_service: MagicMock) -> None:
    store = InMemoryVectorStore(dimensions=3)
    c1 = DocumentChunk(id="c1", content="Match", embedding=[1.0, 0.0, 0.0])
    c2 = DocumentChunk(id="c2", content="Other", embedding=[0.0, 1.0, 0.0])
    await store.upsert([c1, c2])

    service = RetrievalService(store=store, embedding_service=fake_embedding_service)

    response = await service.search(
        RetrievalQuery(
            query="find match",
            top_k=2,
            filters={"foo": "bar"},  # none match foo=bar, will return 0
        )
    )
    assert isinstance(response, RetrievalResponse)
    assert response.total_results == 0

    # Search without filter
    response_no_filter = await service.search(RetrievalQuery(query="find match", top_k=2))
    assert response_no_filter.total_results == 2
    assert response_no_filter.results[0].id == "c1"
    assert response_no_filter.results[0].rank == 1
    assert response_no_filter.query == "find match"
    fake_embedding_service.embed_query.assert_awaited_with("find match")


@pytest.mark.asyncio
async def test_retrieval_service_string_shorthand(fake_embedding_service: MagicMock) -> None:
    store = InMemoryVectorStore(dimensions=3)
    c1 = DocumentChunk(id="c1", content="Match", embedding=[1.0, 0.0, 0.0])
    await store.upsert([c1])

    service = RetrievalService(store=store, embedding_service=fake_embedding_service)
    response = await service.search("shorthand query", top_k=1)
    assert response.total_results == 1
    assert response.results[0].id == "c1"
    assert response.query == "shorthand query"


@pytest.mark.asyncio
async def test_retrieval_service_precomputed_query_vector() -> None:
    # Direct vector search without embedding service
    store = InMemoryVectorStore(dimensions=2)
    c1 = DocumentChunk(id="c1", content="Vector match", embedding=[0.0, 1.0])
    await store.upsert([c1])

    service = RetrievalService(store=store)  # No embedding service provided
    response = await service.search(RetrievalQuery(query_vector=[0.0, 1.0], top_k=5))
    assert response.total_results == 1
    assert response.results[0].id == "c1"


@pytest.mark.asyncio
async def test_retrieval_service_query_vector_dimension_mismatch() -> None:
    store = InMemoryVectorStore(dimensions=3)
    service = RetrievalService(store=store)
    with pytest.raises(IncompatibleEmbeddingError) as exc_info:
        await service.search(RetrievalQuery(query_vector=[1.0, 0.0]))
    assert exc_info.value.requested == 2
    assert exc_info.value.supported == 3


@pytest.mark.asyncio
async def test_retrieval_service_text_query_without_embedding_service_raises() -> None:
    store = InMemoryVectorStore(dimensions=3)
    service = RetrievalService(store=store)
    with pytest.raises(RetrievalError, match="requires an EmbeddingService"):
        await service.search(RetrievalQuery(query="hello"))


def test_retrieval_query_validation() -> None:
    # Both omitted
    with pytest.raises(ValueError, match="Either 'query' .* or 'query_vector'"):
        RetrievalQuery()

    # Empty string
    with pytest.raises(ValueError, match="cannot be empty or whitespace"):
        RetrievalQuery(query="   ")

    # Empty list vector
    with pytest.raises(ValueError, match="Query vector cannot be empty"):
        RetrievalQuery(query_vector=[])

    # Invalid top_k
    with pytest.raises(ValueError):
        RetrievalQuery(query="valid", top_k=0)

    # Invalid score_threshold
    with pytest.raises(ValueError):
        RetrievalQuery(query="valid", score_threshold=1.5)


@pytest.mark.asyncio
async def test_retrieval_service_unsupported_search_mode_rejected() -> None:
    store = InMemoryVectorStore(dimensions=3)
    service = RetrievalService(store=store)

    # Unsupported search mode (e.g. 'graph')
    query_unsupported = RetrievalQuery.model_construct(
        query="test query",
        search_mode="graph",
        top_k=5,
    )
    with pytest.raises(UnsupportedSearchModeError) as exc_info:
        await service.search(query_unsupported)
    assert "Search mode 'graph' is not supported" in str(exc_info.value)
    assert "dense" in exc_info.value.supported_modes
    assert "sparse" in exc_info.value.supported_modes
    assert "hybrid" in exc_info.value.supported_modes


@pytest.mark.asyncio
async def test_retrieval_service_observability_events(fake_embedding_service: MagicMock) -> None:
    store = InMemoryVectorStore(dimensions=3)
    c1 = DocumentChunk(id="c1", content="Match", embedding=[1.0, 0.0, 0.0])
    await store.upsert([c1])
    service = RetrievalService(store=store, embedding_service=fake_embedding_service)

    with patch("genai_core.services.retrieval_service.safe_log") as mock_log:
        await service.search("test obs", top_k=1)

        events_logged = [call.kwargs.get("event") for call in mock_log.call_args_list]
        assert ObservabilityEvent.RETRIEVAL_START in events_logged
        assert ObservabilityEvent.RETRIEVAL_SUCCESS in events_logged


@pytest.mark.asyncio
async def test_retrieval_service_timeout_handling() -> None:
    # Mock store that hangs longer than timeout
    store = MagicMock()
    store.dimensions = 3

    async def _slow_search(*args, **kwargs):
        await asyncio.sleep(0.5)
        return []

    store.search = AsyncMock(side_effect=_slow_search)

    service = RetrievalService(store=store)
    service.retry_policy.timeout_seconds = 0.05
    service.retry_policy.max_retries = 0

    with pytest.raises(RetrievalTimeoutError, match="timed out"):
        await service.search(RetrievalQuery(query_vector=[1.0, 0.0, 0.0]))


@pytest.mark.asyncio
async def test_retrieval_service_index_and_delete_documents(
    fake_embedding_service: MagicMock,
) -> None:
    store = InMemoryVectorStore(dimensions=3)
    service = RetrievalService(store=store, embedding_service=fake_embedding_service)

    # 1. Index raw strings
    count = await service.index_documents(["Text doc 1", "Text doc 2"])
    assert count == 2
    fake_embedding_service.embed_documents.assert_awaited_once()

    # Search to verify indexed
    res = await service.search(RetrievalQuery(query_vector=[1.0, 0.0, 0.0], top_k=5))
    assert res.total_results == 2

    # 2. Index DocumentChunk with pre-computed embedding
    c3 = DocumentChunk(id="c3", content="Pre-embedded", embedding=[0.0, 0.0, 1.0])
    count3 = await service.index_documents([c3])
    assert count3 == 1

    # 3. Index DocumentChunk missing embedding (auto-embeds)
    fake_embedding_service.embed_documents.reset_mock()
    fake_embedding_service.embed_documents.return_value = [[0.5, 0.5, 0.0]]
    c4 = DocumentChunk(id="c4", content="Needs embedding")
    count4 = await service.index_documents([c4])
    assert count4 == 1
    assert c4.embedding == [0.5, 0.5, 0.0]
    fake_embedding_service.embed_documents.assert_awaited_once()

    # 4. Delete document
    deleted = await service.delete_documents(["c3", "c4"])
    assert deleted == 2

    # Empty operations
    assert await service.index_documents([]) == 0
    assert await service.delete_documents([]) == 0


@pytest.mark.asyncio
async def test_retrieval_service_index_documents_without_embedding_service_raises() -> None:
    store = InMemoryVectorStore(dimensions=3)
    service = RetrievalService(store=store)  # No embedding service

    # Indexing raw strings fails
    with pytest.raises(RetrievalError, match="requires an EmbeddingService"):
        await service.index_documents(["Some text"])

    # Indexing DocumentChunk without embedding fails
    c1 = DocumentChunk(id="c1", content="No embedding")
    with pytest.raises(RetrievalError, match="missing embeddings without an EmbeddingService"):
        await service.index_documents([c1])


@pytest.mark.asyncio
async def test_retrieval_service_sparse_search(fake_embedding_service: MagicMock) -> None:
    store = InMemoryVectorStore(dimensions=3)
    c1 = DocumentChunk(
        id="c1",
        content="Postgres full-text search using ts_rank_cd.",
        embedding=[1.0, 0.0, 0.0],
    )
    c2 = DocumentChunk(
        id="c2",
        content="Dense vector embeddings in pgvector.",
        embedding=[0.0, 1.0, 0.0],
    )
    await store.upsert([c1, c2])

    service = RetrievalService(store=store, embedding_service=fake_embedding_service)

    # Sparse search should not call embedding_service
    resp = await service.search(RetrievalQuery(query="ts_rank_cd search", search_mode="sparse"))
    assert resp.search_mode == "sparse"
    assert resp.total_results == 1
    assert resp.results[0].id == "c1"
    fake_embedding_service.embed_query.assert_not_called()


@pytest.mark.asyncio
async def test_retrieval_service_hybrid_search(fake_embedding_service: MagicMock) -> None:
    store = InMemoryVectorStore(dimensions=3)
    c1 = DocumentChunk(
        id="c1",
        content="Python asyncio event loop.",
        embedding=[1.0, 0.0, 0.0],
    )
    c2 = DocumentChunk(
        id="c2",
        content="Python asyncio coroutines and tasks.",
        embedding=[0.8, 0.2, 0.0],
    )
    await store.upsert([c1, c2])

    fake_embedding_service.embed_query.return_value = [1.0, 0.0, 0.0]
    service = RetrievalService(store=store, embedding_service=fake_embedding_service)

    resp = await service.search(
        RetrievalQuery(
            query="coroutines event loop",
            search_mode="hybrid",
            vector_weight=0.5,
            keyword_weight=0.5,
            top_k=2,
        )
    )
    assert resp.search_mode == "hybrid"
    assert resp.total_results == 2
    assert resp.results[0].rank == 1
    fake_embedding_service.embed_query.assert_awaited_once_with("coroutines event loop")


@pytest.mark.asyncio
async def test_retrieval_service_reranking_flow(fake_embedding_service: MagicMock) -> None:
    store = InMemoryVectorStore(dimensions=3)
    c1 = DocumentChunk(id="c1", content="Chunk 1", embedding=[1.0, 0.0, 0.0])
    c2 = DocumentChunk(id="c2", content="Chunk 2", embedding=[0.9, 0.1, 0.0])
    await store.upsert([c1, c2])

    mock_reranker = MagicMock()

    # Reranker reverses the order: c2 becomes rank 1
    async def _mock_rerank(q: str, results: list[Any], top_k: int) -> list[Any]:
        reversed_res = list(reversed(results))[:top_k]
        for idx, r in enumerate(reversed_res, 1):
            r.rank = idx
            r.score = 0.99 if idx == 1 else 0.11
        return reversed_res

    mock_reranker.rerank = _mock_rerank

    service = RetrievalService(
        store=store,
        embedding_service=fake_embedding_service,
        reranker=mock_reranker,
    )

    resp = await service.search(
        RetrievalQuery(
            query="test query",
            query_vector=[1.0, 0.0, 0.0],
            search_mode="dense",
            rerank=True,
            rerank_top_k=2,
        )
    )

    assert resp.reranked is True
    assert resp.results[0].id == "c2"
    assert resp.results[0].score == 0.99


@pytest.mark.asyncio
async def test_retrieval_service_rerank_unavailable_fallback() -> None:
    store = InMemoryVectorStore(dimensions=3)
    c1 = DocumentChunk(id="c1", content="Chunk 1", embedding=[1.0, 0.0, 0.0])
    await store.upsert([c1])

    # No reranker configured, warn_and_continue (default)
    service = RetrievalService(store=store)

    resp = await service.search(
        RetrievalQuery(
            query="test",
            query_vector=[1.0, 0.0, 0.0],
            rerank=True,
            on_rerank_unavailable="warn_and_continue",
        )
    )
    assert resp.reranked is False
    assert resp.rerank_message is not None
    assert len(resp.results) == 1

    # on_rerank_unavailable="raise"
    with pytest.raises(RetrievalError, match="no reranker is configured"):
        await service.search(
            RetrievalQuery(
                query="test",
                query_vector=[1.0, 0.0, 0.0],
                rerank=True,
                on_rerank_unavailable="raise",
            )
        )
