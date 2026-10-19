from __future__ import annotations

import pytest

from genai_core.exceptions import IncompatibleEmbeddingError
from genai_core.models import DocumentChunk
from genai_core.retrieval import InMemoryVectorStore, VectorStoreAdapter


def test_in_memory_store_protocol_conformance() -> None:
    store = InMemoryVectorStore(dimensions=3)
    assert isinstance(store, VectorStoreAdapter)
    assert store.dimensions == 3


def test_in_memory_store_invalid_dimension() -> None:
    with pytest.raises(ValueError, match="dimensions must be greater than 0"):
        InMemoryVectorStore(dimensions=0)


@pytest.mark.asyncio
async def test_in_memory_store_upsert_validation() -> None:
    store = InMemoryVectorStore(dimensions=3)

    # Missing embedding
    chunk_no_emb = DocumentChunk(id="c1", content="Hello world")
    with pytest.raises(ValueError, match="embedding vector is None"):
        await store.upsert([chunk_no_emb])

    # Dimension mismatch
    chunk_wrong_dim = DocumentChunk(id="c2", content="Hello world", embedding=[1.0, 0.0])
    with pytest.raises(IncompatibleEmbeddingError) as exc_info:
        await store.upsert([chunk_wrong_dim])
    assert exc_info.value.requested == 2
    assert exc_info.value.supported == 3

    # Valid upsert
    chunk_valid = DocumentChunk(id="c3", content="Valid", embedding=[1.0, 0.0, 0.0])
    count = await store.upsert([chunk_valid])
    assert count == 1


@pytest.mark.asyncio
async def test_in_memory_store_upsert_and_delete() -> None:
    store = InMemoryVectorStore(dimensions=2)
    c1 = DocumentChunk(id="c1", content="First", embedding=[1.0, 0.0])
    c2 = DocumentChunk(id="c2", content="Second", embedding=[0.0, 1.0])

    assert await store.upsert([c1, c2]) == 2
    # Update existing
    c1_updated = DocumentChunk(id="c1", content="First Updated", embedding=[0.5, 0.5])
    assert await store.upsert([c1_updated]) == 1

    # Search to verify updated content
    results = await store.search(query_vector=[0.5, 0.5], top_k=5)
    assert len(results) == 2
    assert results[0].id == "c1"
    assert results[0].content == "First Updated"

    # Delete
    deleted_count = await store.delete(["c1", "nonexistent"])
    assert deleted_count == 1
    remaining = await store.search(query_vector=[0.0, 1.0], top_k=5)
    assert len(remaining) == 1
    assert remaining[0].id == "c2"


@pytest.mark.asyncio
async def test_in_memory_store_cosine_search_ranking() -> None:
    store = InMemoryVectorStore(dimensions=2)

    # c1 is perfectly aligned with [1.0, 0.0] -> sim = 1.0
    # c2 is at 45 degrees -> sim ~ 0.707
    # c3 is orthogonal [0.0, 1.0] -> sim = 0.0
    c1 = DocumentChunk(id="c1", content="Aligned", embedding=[1.0, 0.0])
    c2 = DocumentChunk(id="c2", content="Diagonal", embedding=[1.0, 1.0])
    c3 = DocumentChunk(id="c3", content="Orthogonal", embedding=[0.0, 1.0])

    await store.upsert([c1, c2, c3])

    results = await store.search(query_vector=[1.0, 0.0], top_k=10)
    assert len(results) == 3
    assert results[0].id == "c1"
    assert results[0].rank == 1
    assert pytest.approx(results[0].score, 0.001) == 1.0

    assert results[1].id == "c2"
    assert results[1].rank == 2
    assert pytest.approx(results[1].score, 0.001) == 0.7071

    assert results[2].id == "c3"
    assert results[2].rank == 3
    assert pytest.approx(results[2].score, 0.001) == 0.0


@pytest.mark.asyncio
async def test_in_memory_store_metadata_filtering() -> None:
    store = InMemoryVectorStore(dimensions=2)

    c1 = DocumentChunk(
        id="c1",
        content="Tenant A Doc",
        embedding=[1.0, 0.0],
        metadata={"tenant_id": "tenant_1", "doc_type": "policy"},
    )
    c2 = DocumentChunk(
        id="c2",
        content="Tenant B Doc",
        embedding=[1.0, 0.0],
        metadata={"tenant_id": "tenant_2", "doc_type": "policy"},
    )
    c3 = DocumentChunk(
        id="c3",
        content="Tenant A FAQ",
        embedding=[1.0, 0.0],
        metadata={"tenant_id": "tenant_1", "doc_type": "faq"},
    )

    await store.upsert([c1, c2, c3])

    # Filter by tenant_id
    res_t1 = await store.search(
        query_vector=[1.0, 0.0],
        top_k=5,
        filters={"tenant_id": "tenant_1"},
    )
    assert len(res_t1) == 2
    assert {r.id for r in res_t1} == {"c1", "c3"}

    # Multi-field filter
    res_t1_policy = await store.search(
        query_vector=[1.0, 0.0],
        top_k=5,
        filters={"tenant_id": "tenant_1", "doc_type": "policy"},
    )
    assert len(res_t1_policy) == 1
    assert res_t1_policy[0].id == "c1"

    # Filter with no matches
    res_none = await store.search(
        query_vector=[1.0, 0.0],
        top_k=5,
        filters={"tenant_id": "nonexistent"},
    )
    assert len(res_none) == 0


@pytest.mark.asyncio
async def test_in_memory_store_score_threshold_and_top_k() -> None:
    store = InMemoryVectorStore(dimensions=2)
    c1 = DocumentChunk(id="c1", content="High", embedding=[1.0, 0.0])
    c2 = DocumentChunk(id="c2", content="Mid", embedding=[1.0, 1.0])
    c3 = DocumentChunk(id="c3", content="Low", embedding=[0.0, 1.0])
    await store.upsert([c1, c2, c3])

    # Top-k limit
    res_top1 = await store.search(query_vector=[1.0, 0.0], top_k=1)
    assert len(res_top1) == 1
    assert res_top1[0].id == "c1"

    # Score threshold filtering: only score >= 0.8 should match
    res_thresh = await store.search(
        query_vector=[1.0, 0.0],
        top_k=5,
        score_threshold=0.8,
    )
    assert len(res_thresh) == 1
    assert res_thresh[0].id == "c1"


@pytest.mark.asyncio
async def test_in_memory_store_search_dimension_mismatch() -> None:
    store = InMemoryVectorStore(dimensions=3)
    with pytest.raises(IncompatibleEmbeddingError) as exc_info:
        await store.search(query_vector=[1.0, 0.0], top_k=5)
    assert exc_info.value.requested == 2
    assert exc_info.value.supported == 3
