from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from semantic_search_eng.models.retrive_response import RetriveResult
from semantic_search_eng.services.retrive_services import RetriveServiceManager


@pytest.fixture
def mock_dependencies():
    return {
        "data_manager": MagicMock(),
        "data_processor": MagicMock(),
        "retrive_manager": AsyncMock(),
        "retrive_db_manager": AsyncMock(),
    }


@pytest.fixture
def retrive_service_manager(monkeypatch, mock_dependencies):
    with (
        patch(
            "semantic_search_eng.services.retrive_services.DataManager",
            return_value=mock_dependencies["data_manager"],
        ),
        patch(
            "semantic_search_eng.services.retrive_services.DataProcessor",
            return_value=mock_dependencies["data_processor"],
        ),
        patch(
            "semantic_search_eng.services.retrive_services.RetriverManager",
            return_value=mock_dependencies["retrive_manager"],
        ),
        patch(
            "semantic_search_eng.services.retrive_services.RetriveDBManager",
            return_value=mock_dependencies["retrive_db_manager"],
        ),
    ):
        manager = RetriveServiceManager()

    return manager, mock_dependencies


@pytest.fixture
def sample_results():
    return [
        RetriveResult(chunk_text="First relevant result", similarity_score=0.95),
        RetriveResult(chunk_text="Second relevant result", similarity_score=0.85),
    ]


@pytest.mark.asyncio
async def test_retrive_chunks_returns_response(retrive_service_manager, sample_results):
    manager, mocks = retrive_service_manager

    mocks["retrive_manager"].retrieve.return_value = sample_results

    from semantic_search_eng.models.retrive_request import RetriveRequest

    request = RetriveRequest(
        tenant_id="550e8400-e29b-41d4-a716-446655440001",
        query="test query",
        top_k=2,
    )

    response = await manager.retrive_chunks(request)

    assert response.tenant_id == request.tenant_id
    assert response.query == request.query
    assert response.top_k == request.top_k
    assert len(response.results) == len(sample_results)
    assert response.results[0].chunk_text == "First relevant result"
    assert response.results[0].similarity_score == 0.95


@pytest.mark.asyncio
async def test_retrive_chunks_calls_retriver_manager(retrive_service_manager, sample_results):
    manager, mocks = retrive_service_manager

    mocks["retrive_manager"].retrieve.return_value = sample_results

    from semantic_search_eng.models.retrive_request import RetriveRequest

    request = RetriveRequest(
        tenant_id="550e8400-e29b-41d4-a716-446655440001",
        query="test query",
        top_k=5,
    )

    await manager.retrive_chunks(request)

    mocks["retrive_manager"].retrieve.assert_called_once()
    call_kwargs = mocks["retrive_manager"].retrieve.call_args[1]
    assert call_kwargs["tenant_id"] == "550e8400-e29b-41d4-a716-446655440001"
    assert call_kwargs["query"] == "test query"
    assert call_kwargs["top_k"] == 5
    assert call_kwargs["document_type"] is None


@pytest.mark.asyncio
async def test_retrive_chunks_calls_retriver_manager_with_document_type(
    retrive_service_manager, sample_results
):
    manager, mocks = retrive_service_manager

    mocks["retrive_manager"].retrieve.return_value = sample_results

    from semantic_search_eng.models.retrive_request import RetriveRequest

    request = RetriveRequest(
        tenant_id="550e8400-e29b-41d4-a716-446655440001",
        query="test query",
        top_k=5,
        document_type="HR Policy",
    )

    await manager.retrive_chunks(request)

    mocks["retrive_manager"].retrieve.assert_called_once()
    call_kwargs = mocks["retrive_manager"].retrieve.call_args[1]
    assert call_kwargs["tenant_id"] == "550e8400-e29b-41d4-a716-446655440001"
    assert call_kwargs["query"] == "test query"
    assert call_kwargs["top_k"] == 5
    assert call_kwargs["document_type"] == "HR Policy"


@pytest.mark.asyncio
async def test_retrive_chunks_returns_empty_results_when_no_matches(retrive_service_manager):
    manager, mocks = retrive_service_manager

    mocks["retrive_manager"].retrieve.return_value = []

    from semantic_search_eng.models.retrive_request import RetriveRequest

    request = RetriveRequest(
        tenant_id="550e8400-e29b-41d4-a716-446655440001",
        query="no results query",
        top_k=5,
    )

    response = await manager.retrive_chunks(request)

    assert len(response.results) == 0
    assert response.tenant_id == request.tenant_id
