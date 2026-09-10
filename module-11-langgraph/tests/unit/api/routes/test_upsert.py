from unittest.mock import AsyncMock
from uuid import uuid4

from fastapi.testclient import TestClient

from rag_app.api.app import app
from rag_app.api.routes import upsert as upsert_route

client = TestClient(app)


def test_upsert_chunks_endpoint_success(monkeypatch):
    mock_service = AsyncMock(return_value=["c_1", "c_2"])
    monkeypatch.setattr(upsert_route.update_service_manager, "upsert_chunks", mock_service)
    payload = {
        "tenant_id": str(uuid4()),
        "document_ids": [str(uuid4()), str(uuid4())],
        "chunk_ids": ["c_1", "c_2"],
        "updated_chunks": ["text 1", "text 2"],
    }
    response = client.post("/upsert/chunks", json=payload)
    assert response.status_code == 200
    assert response.json() == ["c_1", "c_2"]
    mock_service.assert_awaited_once()


def test_upsert_chunks_endpoint_internal_error(monkeypatch):
    mock_service = AsyncMock(side_effect=RuntimeError("Embedding service unavailable"))
    monkeypatch.setattr(upsert_route.update_service_manager, "upsert_chunks", mock_service)
    payload = {
        "tenant_id": str(uuid4()),
        "document_ids": [str(uuid4())],
        "chunk_ids": ["c_1"],
        "updated_chunks": ["text 1"],
    }
    response = client.post("/upsert/chunks", json=payload)
    assert response.status_code == 500
    assert "Embedding service unavailable" in response.json()["detail"]
