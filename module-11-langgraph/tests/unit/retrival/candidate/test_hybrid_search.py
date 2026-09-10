from unittest.mock import AsyncMock, MagicMock

import pytest

from rag_app.models import RetriveResult
from rag_app.retrieval.candidate.hybrid_search import HybridSearch


@pytest.fixture
def mock_settings():
    settings = MagicMock()

    settings.default_vector_search_weight = 0.6
    settings.default_keyword_search_weight = 0.4

    return settings


@pytest.fixture
def hybrid_search(monkeypatch, mock_settings):
    monkeypatch.setattr(
        "rag_app.retrieval.candidate.hybrid_search.get_settings",
        lambda: mock_settings,
    )

    search = HybridSearch()

    search.vector_search.retrive = AsyncMock()
    search.keyword_search.retrive = AsyncMock()

    return search


@pytest.mark.asyncio
async def test_hybrid_search_calls_vector_and_keyword_search(
    hybrid_search,
    tenant_id,
):
    hybrid_search.vector_search.retrive.return_value = []
    hybrid_search.keyword_search.retrive.return_value = []

    queries = [
        "employee leave policy",
        "HR annual leave",
    ]

    await hybrid_search.retrive(
        tenant_id=tenant_id,
        queries=queries,
        top_k_candidates=5,
    )

    hybrid_search.vector_search.retrive.assert_awaited_once_with(
        tenant_id=tenant_id,
        queries=queries,
        top_k_candidates=5,
    )

    hybrid_search.keyword_search.retrive.assert_awaited_once_with(
        tenant_id=tenant_id,
        queries=queries,
        top_k_candidates=5,
    )


def test_hybrid_search_normalizes_scores(
    hybrid_search,
):
    results = [
        RetriveResult(
            chunk_id="chunk_1",
            chunk_text="Chunk 1",
            similarity_score=0.2,
        ),
        RetriveResult(
            chunk_id="chunk_2",
            chunk_text="Chunk 2",
            similarity_score=0.5,
        ),
        RetriveResult(
            chunk_id="chunk_3",
            chunk_text="Chunk 3",
            similarity_score=0.8,
        ),
    ]

    normalized = hybrid_search._normalize_scores(results)

    assert len(normalized) == 3

    assert normalized[0].similarity_score == pytest.approx(0.0)
    assert normalized[1].similarity_score == pytest.approx(0.5)
    assert normalized[2].similarity_score == pytest.approx(1.0)


def test_hybrid_search_normalizes_identical_scores(
    hybrid_search,
):
    results = [
        RetriveResult(
            chunk_id="chunk_1",
            chunk_text="Chunk 1",
            similarity_score=0.8,
        ),
        RetriveResult(
            chunk_id="chunk_2",
            chunk_text="Chunk 2",
            similarity_score=0.8,
        ),
    ]

    normalized = hybrid_search._normalize_scores(results)

    assert len(normalized) == 2

    assert normalized[0].similarity_score == 1.0
    assert normalized[1].similarity_score == 1.0


def test_hybrid_search_normalizes_empty_results(
    hybrid_search,
):
    normalized = hybrid_search._normalize_scores([])

    assert normalized == []


def test_hybrid_search_applies_weights(
    hybrid_search,
):
    vector_results = [
        RetriveResult(
            chunk_id="chunk_1",
            chunk_text="Employee policy",
            similarity_score=1.0,
        )
    ]

    keyword_results = [
        RetriveResult(
            chunk_id="chunk_1",
            chunk_text="Employee policy",
            similarity_score=0.5,
        )
    ]

    merged = hybrid_search._merge_results(
        vector_results=vector_results,
        keyword_results=keyword_results,
    )

    assert len(merged) == 1

    result = merged["chunk_1"]

    expected_score = (1.0 * 0.6) + (0.5 * 0.4)

    assert result.similarity_score == pytest.approx(expected_score)


def test_hybrid_search_keeps_vector_only_results(
    hybrid_search,
):
    vector_results = [
        RetriveResult(
            chunk_id="vector_only",
            chunk_text="Vector result",
            similarity_score=1.0,
        )
    ]

    keyword_results = []

    merged = hybrid_search._merge_results(
        vector_results=vector_results,
        keyword_results=keyword_results,
    )

    assert "vector_only" in merged

    assert merged["vector_only"].similarity_score == pytest.approx(0.6)


def test_hybrid_search_keeps_keyword_only_results(
    hybrid_search,
):
    vector_results = []

    keyword_results = [
        RetriveResult(
            chunk_id="keyword_only",
            chunk_text="Keyword result",
            similarity_score=1.0,
        )
    ]

    merged = hybrid_search._merge_results(
        vector_results=vector_results,
        keyword_results=keyword_results,
    )

    assert "keyword_only" in merged

    assert merged["keyword_only"].similarity_score == pytest.approx(0.4)


def test_hybrid_search_merges_same_chunk(
    hybrid_search,
):
    vector_results = [
        RetriveResult(
            chunk_id="chunk_1",
            chunk_text="Employee leave policy",
            similarity_score=0.8,
        )
    ]

    keyword_results = [
        RetriveResult(
            chunk_id="chunk_1",
            chunk_text="Employee leave policy",
            similarity_score=0.9,
        )
    ]

    merged = hybrid_search._merge_results(
        vector_results=vector_results,
        keyword_results=keyword_results,
    )

    assert len(merged) == 1

    expected_score = (0.8 * 0.6) + (0.9 * 0.4)

    assert merged["chunk_1"].similarity_score == pytest.approx(expected_score)


@pytest.mark.asyncio
async def test_hybrid_search_ranks_results_using_weighted_scores(
    hybrid_search,
    tenant_id,
):
    hybrid_search.vector_search.retrive.return_value = [
        RetriveResult(
            chunk_id="chunk_a",
            chunk_text="Semantic result",
            similarity_score=0.9,
        ),
        RetriveResult(
            chunk_id="chunk_b",
            chunk_text="Keyword-focused result",
            similarity_score=0.6,
        ),
    ]

    hybrid_search.keyword_search.retrive.return_value = [
        RetriveResult(
            chunk_id="chunk_a",
            chunk_text="Semantic result",
            similarity_score=0.4,
        ),
        RetriveResult(
            chunk_id="chunk_b",
            chunk_text="Keyword-focused result",
            similarity_score=1.0,
        ),
    ]

    results = await hybrid_search.retrive(
        tenant_id=tenant_id,
        queries=["employee leave"],
        top_k_candidates=2,
    )

    assert len(results) == 2

    assert results[0].chunk_id == "chunk_a"
    assert results[1].chunk_id == "chunk_b"

    assert results[0].similarity_score == pytest.approx(0.6)
    assert results[1].similarity_score == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_hybrid_search_respects_top_k(
    hybrid_search,
    tenant_id,
):
    hybrid_search.vector_search.retrive.return_value = [
        RetriveResult(
            chunk_id="chunk_1",
            chunk_text="Chunk 1",
            similarity_score=0.9,
        ),
        RetriveResult(
            chunk_id="chunk_2",
            chunk_text="Chunk 2",
            similarity_score=0.8,
        ),
        RetriveResult(
            chunk_id="chunk_3",
            chunk_text="Chunk 3",
            similarity_score=0.7,
        ),
    ]

    hybrid_search.keyword_search.retrive.return_value = []

    results = await hybrid_search.retrive(
        tenant_id=tenant_id,
        queries=["employee"],
        top_k_candidates=2,
    )

    assert len(results) == 2

    assert [result.chunk_id for result in results] == [
        "chunk_1",
        "chunk_2",
    ]


@pytest.mark.asyncio
async def test_hybrid_search_handles_empty_vector_results(
    hybrid_search,
    tenant_id,
):
    hybrid_search.vector_search.retrive.return_value = []

    hybrid_search.keyword_search.retrive.return_value = [
        RetriveResult(
            chunk_id="chunk_1",
            chunk_text="Keyword result",
            similarity_score=0.9,
        )
    ]

    results = await hybrid_search.retrive(
        tenant_id=tenant_id,
        queries=["employee"],
        top_k_candidates=5,
    )

    assert len(results) == 1

    assert results[0].chunk_id == "chunk_1"


@pytest.mark.asyncio
async def test_hybrid_search_handles_empty_keyword_results(
    hybrid_search,
    tenant_id,
):
    hybrid_search.vector_search.retrive.return_value = [
        RetriveResult(
            chunk_id="chunk_1",
            chunk_text="Vector result",
            similarity_score=0.9,
        )
    ]

    hybrid_search.keyword_search.retrive.return_value = []

    results = await hybrid_search.retrive(
        tenant_id=tenant_id,
        queries=["employee"],
        top_k_candidates=5,
    )

    assert len(results) == 1

    assert results[0].chunk_id == "chunk_1"


@pytest.mark.asyncio
async def test_hybrid_search_handles_no_results(
    hybrid_search,
    tenant_id,
):
    hybrid_search.vector_search.retrive.return_value = []
    hybrid_search.keyword_search.retrive.return_value = []

    results = await hybrid_search.retrive(
        tenant_id=tenant_id,
        queries=["unknown query"],
        top_k_candidates=5,
    )

    assert results == []


@pytest.mark.asyncio
async def test_hybrid_search_weight_changes_affect_ranking(
    hybrid_search,
    tenant_id,
):
    hybrid_search.vector_search.retrive.return_value = [
        RetriveResult(
            chunk_id="vector_favorite",
            chunk_text="Semantic result",
            similarity_score=0.9,
        ),
        RetriveResult(
            chunk_id="keyword_favorite",
            chunk_text="Exact keyword result",
            similarity_score=0.6,
        ),
    ]

    hybrid_search.keyword_search.retrive.return_value = [
        RetriveResult(
            chunk_id="vector_favorite",
            chunk_text="Semantic result",
            similarity_score=0.4,
        ),
        RetriveResult(
            chunk_id="keyword_favorite",
            chunk_text="Exact keyword result",
            similarity_score=1.0,
        ),
    ]

    # Vector-heavy configuration
    hybrid_search.settings.default_vector_search_weight = 0.8
    hybrid_search.settings.default_keyword_search_weight = 0.2

    vector_heavy_results = await hybrid_search.retrive(
        tenant_id=tenant_id,
        queries=["employee leave"],
        top_k_candidates=2,
    )

    assert vector_heavy_results[0].chunk_id == "vector_favorite"

    # Keyword-heavy configuration
    hybrid_search.settings.default_vector_search_weight = 0.2
    hybrid_search.settings.default_keyword_search_weight = 0.8

    keyword_heavy_results = await hybrid_search.retrive(
        tenant_id=tenant_id,
        queries=["employee leave"],
        top_k_candidates=2,
    )

    assert keyword_heavy_results[0].chunk_id == "keyword_favorite"
