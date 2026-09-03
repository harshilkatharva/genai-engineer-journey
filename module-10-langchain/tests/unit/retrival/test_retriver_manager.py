from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from langchain_core.documents import Document

from rag_app.models import RetriveRequest, RetriveResponse, RetriveResult
from rag_app.retrieval.retriver_manager import LangchainRetriever, RetriverManager


@pytest.mark.asyncio
async def test_retrieve_returns_results():
    tenant_id = uuid4()
    queries = ["What is the refund policy?"]
    top_k_candidate = 5

    expected_results = [
        RetriveResult(
            chunk_id="chunk_text_01",
            chunk_text="Customers can request a refund within 30 days.",
            similarity_score=0.92,
        ),
        RetriveResult(
            chunk_id="chunk_text_02",
            chunk_text="Refund requests must include the original receipt.",
            similarity_score=0.85,
        ),
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
            chunk_id="test_chunk",
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

    vector_search.retrive.assert_awaited_once_with(
        tenant_id=tenant_id,
        queries=queries,
        top_k_candidates=5,
    )


@pytest.mark.asyncio
async def test_retrieve_uses_vector_search_strategy():
    tenant_id = uuid4()
    queries = ["employee leave policy"]

    expected_results = [
        RetriveResult(
            chunk_id="vector_chunk",
            chunk_text="Employee leave policy",
            similarity_score=0.91,
        )
    ]

    with (
        patch("rag_app.retrieval.retriver_manager.get_settings") as mock_get_settings,
        patch("rag_app.retrieval.retriver_manager.VectorSearch") as mock_vector_search,
        patch("rag_app.retrieval.retriver_manager.KeywordSearch") as mock_keyword_search,
        patch("rag_app.retrieval.retriver_manager.HybridSearch") as mock_hybrid_search,
    ):
        settings = MagicMock()
        settings.default_retrieval_strategy = "vector_search"
        settings.re_ranker_availability = False
        mock_get_settings.return_value = settings

        vector_search = mock_vector_search.return_value
        vector_search.retrive = AsyncMock(return_value=expected_results)

        keyword_search = mock_keyword_search.return_value
        hybrid_search = mock_hybrid_search.return_value

        manager = RetriverManager()

        request = RetriveRequest(
            tenant_id=tenant_id,
            queries=queries,
            top_k_candidate=5,
        )

        response = await manager.retrieve(request)

    assert response.results == expected_results

    vector_search.retrive.assert_awaited_once_with(
        tenant_id=tenant_id,
        queries=queries,
        top_k_candidates=5,
    )

    keyword_search.retrive.assert_not_called()
    hybrid_search.retrive.assert_not_called()


@pytest.mark.asyncio
async def test_retrieve_uses_keyword_search_strategy():
    tenant_id = uuid4()
    queries = ["EMP-1024 payroll process"]

    expected_results = [
        RetriveResult(
            chunk_id="keyword_chunk",
            chunk_text="EMP-1024 payroll process",
            similarity_score=0.88,
        )
    ]

    with (
        patch("rag_app.retrieval.retriver_manager.get_settings") as mock_get_settings,
        patch("rag_app.retrieval.retriver_manager.VectorSearch") as mock_vector_search,
        patch("rag_app.retrieval.retriver_manager.KeywordSearch") as mock_keyword_search,
        patch("rag_app.retrieval.retriver_manager.HybridSearch") as mock_hybrid_search,
    ):
        settings = MagicMock()
        settings.default_retrieval_strategy = "keyword_search"
        settings.re_ranker_availability = False
        mock_get_settings.return_value = settings

        vector_search = mock_vector_search.return_value

        keyword_search = mock_keyword_search.return_value
        keyword_search.retrive = AsyncMock(return_value=expected_results)

        hybrid_search = mock_hybrid_search.return_value

        manager = RetriverManager()

        request = RetriveRequest(
            tenant_id=tenant_id,
            queries=queries,
            top_k_candidate=5,
        )

        response = await manager.retrieve(request)

    assert response.results == expected_results

    keyword_search.retrive.assert_awaited_once_with(
        tenant_id=tenant_id,
        queries=queries,
        top_k_candidates=5,
    )

    vector_search.retrive.assert_not_called()
    hybrid_search.retrive.assert_not_called()


@pytest.mark.asyncio
async def test_retrieve_uses_hybrid_search_strategy():
    tenant_id = uuid4()
    queries = ["How do I reset an employee password?"]

    expected_results = [
        RetriveResult(
            chunk_id="hybrid_chunk_1",
            chunk_text="Employee password reset process",
            similarity_score=0.94,
        ),
        RetriveResult(
            chunk_id="hybrid_chunk_2",
            chunk_text="Password reset policy",
            similarity_score=0.82,
        ),
    ]

    with (
        patch("rag_app.retrieval.retriver_manager.get_settings") as mock_get_settings,
        patch("rag_app.retrieval.retriver_manager.VectorSearch") as mock_vector_search,
        patch("rag_app.retrieval.retriver_manager.KeywordSearch") as mock_keyword_search,
        patch("rag_app.retrieval.retriver_manager.HybridSearch") as mock_hybrid_search,
    ):
        settings = MagicMock()
        settings.default_retrieval_strategy = "hybrid_search"
        settings.re_ranker_availability = False
        mock_get_settings.return_value = settings

        vector_search = mock_vector_search.return_value
        keyword_search = mock_keyword_search.return_value

        hybrid_search = mock_hybrid_search.return_value
        hybrid_search.retrive = AsyncMock(return_value=expected_results)

        manager = RetriverManager()

        request = RetriveRequest(
            tenant_id=tenant_id,
            queries=queries,
            top_k_candidate=5,
        )

        response = await manager.retrieve(request)

    assert response.results == expected_results

    hybrid_search.retrive.assert_awaited_once_with(
        tenant_id=tenant_id,
        queries=queries,
        top_k_candidates=5,
    )

    vector_search.retrive.assert_not_called()
    keyword_search.retrive.assert_not_called()


@pytest.mark.asyncio
async def test_langchain_retriever_aget_relevant_documents():
    tenant_id = uuid4()
    queries = ["What is the refund policy?"]
    chunk_id_1 = str(uuid4())
    chunk_id_2 = str(uuid4())

    mock_retrive_results = [
        RetriveResult(
            chunk_id=chunk_id_1,
            chunk_text="Customers can request a refund within 30 days.",
            similarity_score=0.92,
        ),
        RetriveResult(
            chunk_id=chunk_id_2,
            chunk_text="Refund requests must include original receipt.",
            similarity_score=0.85,
        ),
    ]

    mock_response = RetriveResponse(
        tenant_id=tenant_id,
        queries=queries,
        results=mock_retrive_results,
    )

    mock_manager = MagicMock(spec=RetriverManager)
    mock_manager.retrieve = AsyncMock(return_value=mock_response)

    retriever = LangchainRetriever(retriever_manager=mock_manager)

    docs = await retriever._aget_relevant_documents(
        tenant_id=tenant_id,
        queries=queries,
    )

    mock_manager.retrieve.assert_awaited_once()
    called_request = mock_manager.retrieve.call_args[0][0]
    assert isinstance(called_request, RetriveRequest)
    assert called_request.tenant_id == tenant_id
    assert called_request.queries == queries

    assert len(docs) == 2
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "Customers can request a refund within 30 days."
    assert docs[0].metadata == {
        "chunk_id": chunk_id_1,
        "similarity_score": 0.92,
    }
    assert isinstance(docs[1], Document)
    assert docs[1].page_content == "Refund requests must include original receipt."
    assert docs[1].metadata == {
        "chunk_id": chunk_id_2,
        "similarity_score": 0.85,
    }


def test_langchain_retriever_get_relevant_documents_raises_not_implemented():
    mock_manager = MagicMock(spec=RetriverManager)
    retriever = LangchainRetriever(retriever_manager=mock_manager)

    with pytest.raises(NotImplementedError) as exc_info:
        retriever._get_relevant_documents("test query")

    assert "This retriever supports async retrieval. Use ainvoke()." in str(exc_info.value)
