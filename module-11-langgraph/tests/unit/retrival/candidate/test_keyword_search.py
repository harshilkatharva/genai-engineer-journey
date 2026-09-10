from unittest.mock import AsyncMock

import pytest

from rag_app.models import RetriveResult
from rag_app.retrieval.candidate.keyword_search import KeywordSearch


@pytest.mark.asyncio
async def test_keyword_search_retrieves_and_deduplicates_results():
    tenant_id = "5c664d1a-616d-46a4-9ef3-d3934166a1"

    search = KeywordSearch()

    search.retrive_db_manager.retrive_keyword_chunks = AsyncMock(
        side_effect=[
            [
                RetriveResult(
                    chunk_id="chunk-1",
                    chunk_text="Employee leave policy",
                    similarity_score=0.80,
                ),
                RetriveResult(
                    chunk_id="chunk-2",
                    chunk_text="HR leave approval",
                    similarity_score=0.60,
                ),
            ],
            [
                RetriveResult(
                    chunk_id="chunk-1",
                    chunk_text="Employee leave policy",
                    similarity_score=0.90,
                ),
                RetriveResult(
                    chunk_id="chunk-3",
                    chunk_text="Leave request process",
                    similarity_score=0.70,
                ),
            ],
        ]
    )

    results = await search.retrive(
        tenant_id=tenant_id,
        queries=[
            "employee leave policy",
            "leave approval process",
        ],
        top_k_candidates=3,
    )

    assert len(results) == 3

    assert [result.chunk_id for result in results] == [
        "chunk-1",
        "chunk-3",
        "chunk-2",
    ]

    assert results[0].similarity_score == 0.90

    assert search.retrive_db_manager.retrive_keyword_chunks.await_count == 2


@pytest.mark.asyncio
async def test_keyword_search_limits_final_results():
    tenant_id = "5c664d1a-616d-46a4-9ef3-d3934166a1"

    search = KeywordSearch()

    search.retrive_db_manager.retrive_keyword_chunks = AsyncMock(
        return_value=[
            RetriveResult(
                chunk_id="chunk-1",
                chunk_text="first",
                similarity_score=0.90,
            ),
            RetriveResult(
                chunk_id="chunk-2",
                chunk_text="third",
                similarity_score=0.70,
            ),
            RetriveResult(
                chunk_id="chunk-3",
                chunk_text="second",
                similarity_score=0.80,
            ),
        ]
    )

    results = await search.retrive(
        tenant_id=tenant_id,
        queries=["employee policy"],
        top_k_candidates=2,
    )

    assert len(results) == 2

    assert [result.chunk_id for result in results] == [
        "chunk-1",
        "chunk-3",
    ]


@pytest.mark.asyncio
async def test_keyword_search_returns_empty_when_no_matches():
    tenant_id = "5c664d1a-616d-46a4-9ef3-d3934166a1"

    search = KeywordSearch()

    search.retrive_db_manager.retrive_keyword_chunks = AsyncMock(return_value=[])

    results = await search.retrive(
        tenant_id=tenant_id,
        queries=["completely unrelated query"],
        top_k_candidates=10,
    )

    assert results == []

    search.retrive_db_manager.retrive_keyword_chunks.assert_awaited_once()


@pytest.mark.asyncio
async def test_keyword_search_calls_db_for_each_query():
    tenant_id = "5c664d1a-616d-46a4-9ef3-d3934166a1"

    search = KeywordSearch()

    search.retrive_db_manager.retrive_keyword_chunks = AsyncMock(return_value=[])

    queries = [
        "employee leave",
        "HR policy",
        "annual vacation",
    ]

    await search.retrive(
        tenant_id=tenant_id,
        queries=queries,
        top_k_candidates=10,
    )

    assert search.retrive_db_manager.retrive_keyword_chunks.await_count == 3

    calls = search.retrive_db_manager.retrive_keyword_chunks.await_args_list

    assert calls[0].kwargs["query"] == "employee leave"
    assert calls[1].kwargs["query"] == "HR policy"
    assert calls[2].kwargs["query"] == "annual vacation"
