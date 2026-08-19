from unittest.mock import patch
from uuid import uuid4
from fastapi import FastAPI
from fastapi.testclient import TestClient
from rag_app.api.routes.upsert import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)


@patch("rag_app.api.routes.upsert.UpsertServiceManager.upsert_chunks")
def test_upsert_chunks_endpoint_success(mock_service):
    mock_service.return_value = ["c_1", "c_2"]
    payload = {
        "tenant_id": str(uuid4()),
        "document_ids": [str(uuid4()), str(uuid4())],
        "chunk_ids": ["c_1", "c_2"],
        "updated_chunks": ["text 1", "text 2"],
    }
    response = client.post("/chunks", json=payload)
    assert response.status_code == 200
    assert response.json() == ["c_1", "c_2"]


@patch("rag_app.api.routes.upsert.UpsertServiceManager.upsert_chunks")
def test_upsert_chunks_endpoint_internal_error(mock_service):
    mock_service.side_effect = RuntimeError("Embedding service unavailable")
    payload = {
        "tenant_id": str(uuid4()),
        "document_ids": [str(uuid4())],
        "chunk_ids": ["c_1"],
        "updated_chunks": ["text 1"],
    }
    response = client.post("/chunks", json=payload)
    assert response.status_code == 500
    assert "Embedding service unavailable" in response.json()["detail"]
