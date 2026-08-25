from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from rag_app.models import (
    RetriveRequest,
    RetriveResponse,
    RetriveResult,
)
from rag_app.services.retrive_services import RetriveServiceManager


@pytest.fixture
def retrive_service_manager():
    """
    Create RetriveServiceManager with mocked dependencies.
    """
    manager = RetriveServiceManager()

    data_manager = MagicMock()
    retrive_manager = AsyncMock()
    retrive_db_manager = AsyncMock()

    manager.data_manager = data_manager
    manager.retrive_manager = retrive_manager
    manager.retrive_db_manager = retrive_db_manager

    mocks = {
        "data_manager": data_manager,
        "retrive_manager": retrive_manager,
        "retrive_db_manager": retrive_db_manager,
    }

    return manager, mocks


@pytest.mark.asyncio
async def test_retrive_chunks_returns_empty_results_when_no_matches(
    retrive_service_manager,
):
    # Arrange
    manager, mocks = retrive_service_manager

    tenant_id = UUID("550e8400-e29b-41d4-a716-446655440001")

    request = RetriveRequest(
        tenant_id=tenant_id,
        queries=["no results query"],
    )

    # RetriveManager.retrieve() returns RetriveResponse,
    # NOT list[RetriveResult].
    mocks["retrive_manager"].retrieve.return_value = RetriveResponse(
        tenant_id=tenant_id,
        queries=request.queries,
        results=[],
    )

    # Act
    response = await manager.retrive_chunks(request)

    # Assert
    assert isinstance(response, RetriveResponse)
    assert response.tenant_id == tenant_id
    assert response.queries == ["no results query"]
    assert response.results == []

    mocks["retrive_manager"].retrieve.assert_awaited_once_with(request=request)


@pytest.mark.asyncio
async def test_retrive_chunks_returns_results(
    retrive_service_manager,
):
    # Arrange
    manager, mocks = retrive_service_manager

    tenant_id = UUID("550e8400-e29b-41d4-a716-446655440001")

    request = RetriveRequest(
        tenant_id=tenant_id,
        queries=["What is RAG?"],
    )

    results = [
        RetriveResult(
            chunk_id="test_chunk_001",
            chunk_text="RAG stands for Retrieval-Augmented Generation.",
            similarity_score=0.95,
        ),
        RetriveResult(
            chunk_id="test_chunk_002",
            chunk_text="RAG retrieves relevant information before generation.",
            similarity_score=0.89,
        ),
    ]

    mocks["retrive_manager"].retrieve.return_value = RetriveResponse(
        tenant_id=tenant_id,
        queries=request.queries,
        results=results,
    )

    # Act
    response = await manager.retrive_chunks(request)

    # Assert
    assert isinstance(response, RetriveResponse)

    assert response.tenant_id == tenant_id

    assert response.queries == [
        "What is RAG?",
    ]

    assert response.results == results
    assert len(response.results) == 2

    assert response.results[0].chunk_text == ("RAG stands for Retrieval-Augmented Generation.")
    assert response.results[0].similarity_score == 0.95

    assert response.results[1].chunk_text == (
        "RAG retrieves relevant information before generation."
    )
    assert response.results[1].similarity_score == 0.89

    mocks["retrive_manager"].retrieve.assert_awaited_once_with(request=request)


@pytest.mark.asyncio
async def test_retrive_chunks_passes_request_to_retrive_manager(
    retrive_service_manager,
):
    # Arrange
    manager, mocks = retrive_service_manager

    tenant_id = UUID("550e8400-e29b-41d4-a716-446655440001")

    request = RetriveRequest(
        tenant_id=tenant_id,
        queries=[
            "What is RAG?",
            "How does retrieval work?",
        ],
    )

    mocks["retrive_manager"].retrieve.return_value = RetriveResponse(
        tenant_id=tenant_id,
        queries=request.queries,
        results=[],
    )

    # Act
    await manager.retrive_chunks(request)

    # Assert
    mocks["retrive_manager"].retrieve.assert_awaited_once_with(request=request)


@pytest.mark.asyncio
async def test_retrive_chunks_preserves_queries(
    retrive_service_manager,
):
    # Arrange
    manager, mocks = retrive_service_manager

    tenant_id = UUID("550e8400-e29b-41d4-a716-446655440001")

    queries = [
        "first query",
        "second query",
        "third query",
    ]

    request = RetriveRequest(
        tenant_id=tenant_id,
        queries=queries,
    )

    mocks["retrive_manager"].retrieve.return_value = RetriveResponse(
        tenant_id=tenant_id,
        queries=queries,
        results=[],
    )

    # Act
    response = await manager.retrive_chunks(request)

    # Assert
    assert response.queries == queries


@pytest.mark.asyncio
async def test_retrive_chunks_preserves_tenant_id(
    retrive_service_manager,
):
    # Arrange
    manager, mocks = retrive_service_manager

    tenant_id = UUID("550e8400-e29b-41d4-a716-446655440001")

    request = RetriveRequest(
        tenant_id=tenant_id,
        queries=["test query"],
    )

    mocks["retrive_manager"].retrieve.return_value = RetriveResponse(
        tenant_id=tenant_id,
        queries=request.queries,
        results=[],
    )

    # Act
    response = await manager.retrive_chunks(request)

    # Assert
    assert response.tenant_id == tenant_id


@pytest.mark.asyncio
async def test_retrive_chunks_preserves_retrive_results(
    retrive_service_manager,
):
    # Arrange
    manager, mocks = retrive_service_manager

    tenant_id = UUID("550e8400-e29b-41d4-a716-446655440001")

    request = RetriveRequest(
        tenant_id=tenant_id,
        queries=["test query"],
    )

    expected_results = [
        RetriveResult(
            chunk_id="test_chunk_text_1",
            chunk_text="First retrieved chunk",
            similarity_score=0.98,
        ),
        RetriveResult(
            chunk_id="test_chunk_text_1",
            chunk_text="Second retrieved chunk",
            similarity_score=0.91,
        ),
    ]

    mocks["retrive_manager"].retrieve.return_value = RetriveResponse(
        tenant_id=tenant_id,
        queries=request.queries,
        results=expected_results,
    )

    # Act
    response = await manager.retrive_chunks(request)

    # Assert
    assert response.results == expected_results
    assert len(response.results) == 2
