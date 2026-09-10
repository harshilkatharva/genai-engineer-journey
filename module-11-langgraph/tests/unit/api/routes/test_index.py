from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from rag_app.api.app import app
from rag_app.api.routes import index as index_route


client = TestClient(app)


def test_process_documents_valid_request(monkeypatch):
    expected = {
        "tenant_id": "550e8400-e29b-41d4-a716-446655440001",
        "document_count": 1,
        "chunk_count": 1,
        "embedding_count": 1,
        "document_ids": [],
    }
    index = AsyncMock(return_value=expected)
    monkeypatch.setattr(index_route.index_service_manager, "index", index)

    response = client.post(
        "/index/process",
        json={
            "tenant_id": expected["tenant_id"],
            "documents": ["Document 1"],
            "documents_type": ["pdf"],
            "meta_data": [{"page": 1}],
        },
    )

    assert response.status_code == 200
    assert response.json() == expected
    index.assert_awaited_once()


def test_process_documents_service_error(monkeypatch):
    index = AsyncMock(side_effect=RuntimeError("indexing unavailable"))
    monkeypatch.setattr(index_route.index_service_manager, "index", index)

    response = client.post(
        "/index/process",
        json={
            "tenant_id": "550e8400-e29b-41d4-a716-446655440001",
            "documents": ["Document 1"],
            "documents_type": ["pdf"],
            "meta_data": [{"page": 1}],
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "indexing unavailable"


def test_process_documents_rejects_mismatched_arrays():
    response = client.post(
        "/index/process",
        json={
            "tenant_id": "550e8400-e29b-41d4-a716-446655440001",
            "documents": ["Document 1", "Document 2"],
            "documents_type": ["pdf"],
            "meta_data": [{"page": 1}, {"page": 2}],
        },
    )

    assert response.status_code == 422
