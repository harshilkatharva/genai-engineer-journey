from unittest.mock import AsyncMock
from uuid import uuid4

from fastapi.testclient import TestClient

from rag_app.api.app import app
from rag_app.api.routes import retrive as retrive_route
from rag_app.models import RetriveResponse


client = TestClient(app)


def test_retrive_valid_request(monkeypatch):
    tenant_id = uuid4()
    expected = RetriveResponse(tenant_id=tenant_id, queries=["policy"], results=[])
    retrieve = AsyncMock(return_value=expected)
    monkeypatch.setattr(retrive_route.retrive_service_manager, "retrive_chunks", retrieve)

    response = client.post(
        "/retrive/",
        json={"tenant_id": str(tenant_id), "query": "policy"},
    )

    assert response.status_code == 200
    assert response.json() == expected.model_dump(mode="json")
    retrieve.assert_awaited_once()


def test_retrive_value_error_returns_bad_request(monkeypatch):
    retrieve = AsyncMock(side_effect=ValueError("Query cannot be empty."))
    monkeypatch.setattr(retrive_route.retrive_service_manager, "retrive_chunks", retrieve)

    response = client.post(
        "/retrive/",
        json={"tenant_id": str(uuid4()), "query": "query"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Query cannot be empty."


def test_retrive_runtime_error_returns_server_error(monkeypatch):
    retrieve = AsyncMock(side_effect=RuntimeError("database unavailable"))
    monkeypatch.setattr(retrive_route.retrive_service_manager, "retrive_chunks", retrieve)

    response = client.post(
        "/retrive/",
        json={"tenant_id": str(uuid4()), "query": "query"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "database unavailable"


def test_retrive_rejects_invalid_top_k():
    response = client.post(
        "/retrive/",
        json={"tenant_id": str(uuid4()), "query": "query", "top_k_candidate": 0},
    )

    assert response.status_code == 422
