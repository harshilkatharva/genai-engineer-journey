from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from rag_app.models import RetriveRequest, RetriveResult
from rag_app.retrieval.retriver_manager import RetriverManager


@pytest.mark.asyncio
async def test_retrieve_returns_results():
    tenant_id = uuid4()
    queries = ["What is the refund policy?"]
    top_k_candidate = 5

    expected_results = [
        RetriveResult(
            chunk_text="Customers can request a refund within 30 days.",
            similarity_score=0.92,
        ),
        RetriveResult(
            chunk_text="Refund requests must include the original receipt.",
            similarity_score=0.85,
        ),
    ]

    with (
        patch("rag_app.retrieval.retriver_manager.get_settings") as mock_get_settings,
        patch("rag_app.retrieval.retriver_manager.VectorSearch") as mock_vector_search,
    ):
        # Settings
        settings = MagicMock()
        settings.default_retrieval_strategy = "vector_search"
        settings.re_ranker_availability = False
        mock_get_settings.return_value = settings

        # Vector search
        vector_search = mock_vector_search.return_value
        vector_search.retrive = AsyncMock(return_value=expected_results)

        manager = RetriverManager()

        request = RetriveRequest(
            tenant_id=tenant_id,
            queries=queries,
            top_k_candidate=top_k_candidate,
        )

        response = await manager.retrieve(request)

    assert response.tenant_id == tenant_id
    assert response.queries == queries
    assert response.results == expected_results

    vector_search.retrive.assert_awaited_once_with(
        tenant_id=tenant_id,
        queries=queries,
        top_k_candidates=top_k_candidate,
    )


@pytest.mark.asyncio
async def test_retrieve_uses_configured_retrieval_strategy():
    tenant_id = uuid4()
    queries = ["test query"]

    expected_results = [
        RetriveResult(
            chunk_text="test chunk",
            similarity_score=0.9,
        )
    ]

    with (
        patch("rag_app.retrieval.retriver_manager.get_settings") as mock_get_settings,
        patch("rag_app.retrieval.retriver_manager.VectorSearch") as mock_vector_search,
    ):
        settings = MagicMock()
        settings.default_retrieval_strategy = "vector_search"
        settings.re_ranker_availability = False
        mock_get_settings.return_value = settings

        vector_search = mock_vector_search.return_value
        vector_search.retrive = AsyncMock(return_value=expected_results)

        manager = RetriverManager()

        request = RetriveRequest(
            tenant_id=tenant_id,
            queries=queries,
            document_type=["pdf"],
            top_k_candidate=5,
        )

        response = await manager.retrieve(request)

    assert response.results == expected_results
    vector_search.retrive.assert_awaited_once()
