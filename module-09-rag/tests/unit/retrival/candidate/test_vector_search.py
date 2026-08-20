import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, patch

from pgvector import Vector

from rag_app.retrieval.candidate.vector_search import VectorSearch


@pytest.mark.asyncio
async def test_retrive_returns_results_for_each_query():
    tenant_id = uuid4()
    queries = ["What is the refund policy?", "What is the cancellation policy?"]
    top_k_candidates = 5

    query_embedding_1 = [0.1, 0.2, 0.3]
    query_embedding_2 = [0.4, 0.5, 0.6]

    db_result_1 = ["chunk-1", "chunk-2"]
    db_result_2 = ["chunk-3", "chunk-4"]

    with (
        patch("rag_app.retrieval.candidate.vector_search.RetriveDBManager") as mock_db_manager,
        patch(
            "rag_app.retrieval.candidate.vector_search.EmbeddingManager"
        ) as mock_embedding_manager,
    ):
        # Mock embedding manager
        embedding_manager = mock_embedding_manager.return_value
        embedding_manager.embed_query.side_effect = [
            query_embedding_1,
            query_embedding_2,
        ]

        # Mock DB manager
        db_manager = mock_db_manager.return_value
        db_manager.retrive_chunks = AsyncMock(
            side_effect=[
                db_result_1,
                db_result_2,
            ]
        )

        vector_search = VectorSearch()

        results = await vector_search.retrive(
            tenant_id=tenant_id,
            queries=queries,
            top_k_candidates=top_k_candidates,
        )

    assert results == [db_result_1, db_result_2]

    # Embedding generated once per query
    assert embedding_manager.embed_query.call_count == 2
    embedding_manager.embed_query.assert_any_call(queries[0])
    embedding_manager.embed_query.assert_any_call(queries[1])

    # DB called once per query
    assert db_manager.retrive_chunks.call_count == 2

    db_manager.retrive_chunks.assert_any_await(
        tenant_id,
        Vector(query_embedding_1),
        top_k_candidates,
    )

    db_manager.retrive_chunks.assert_any_await(
        tenant_id,
        Vector(query_embedding_2),
        top_k_candidates,
    )
