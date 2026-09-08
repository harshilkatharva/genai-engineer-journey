from uuid import UUID

from fastapi.testclient import TestClient

from rag_app.api.app import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "running"
    assert "name" in body
    assert "version" in body


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_generate_uuid_endpoint():
    response = client.get("/genrate_uuid")

    assert response.status_code == 200
    UUID(response.json())
