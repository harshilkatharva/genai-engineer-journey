from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from rag_app.api.app import create_app
from rag_app.schemas.rag import HealthResponse


def test_create_app_registers_health_and_rag_routes() -> None:
    settings = MagicMock(app_name="test-rag")
    service = MagicMock(ready=False, document_count=0)

    with (
        patch("rag_app.api.app.get_settings", return_value=settings),
        patch("rag_app.api.app.RAGService", return_value=service),
    ):
        app = create_app()

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "ready": False, "documents": 0}
    assert app.title == "test-rag"
    assert app.state.rag_pipeline is service
    assert "/query" in client.get("/openapi.json").json()["paths"]


def test_health_response_uses_schema() -> None:
    assert HealthResponse(status="ok", ready=False, documents=0).model_dump() == {
        "status": "ok",
        "ready": False,
        "documents": 0,
    }
