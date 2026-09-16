from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_app.api.routes.rag import router
from rag_app.schemas.rag import IngestResponse, QueryResponse, SourceReference


def make_client(service: MagicMock) -> TestClient:
    app = FastAPI()
    app.state.rag_pipeline = service
    app.include_router(router)
    return TestClient(app)


def test_ingest_endpoint_delegates_to_service() -> None:
    service = MagicMock()
    service.ingest = AsyncMock(
        return_value=IngestResponse(source_dir="/data", documents=2, nodes=8)
    )

    response = make_client(service).post("/ingest", json={"data_dir": "/data"})

    assert response.status_code == 200
    assert response.json() == {"source_dir": "/data", "documents": 2, "nodes": 8}
    service.ingest.assert_called_once_with("/data")


def test_ingest_endpoint_maps_service_errors() -> None:
    service = MagicMock()
    service.ingest = AsyncMock(side_effect=ValueError("No supported documents"))

    response = make_client(service).post("/ingest", json={})

    assert response.status_code == 400
    assert response.json() == {"detail": "No supported documents"}


def test_query_endpoint_supports_automatic_routing() -> None:
    service = MagicMock()
    service.query = AsyncMock(
        return_value=QueryResponse(
            answer="answer",
            route="summary",
            sources=[SourceReference(source="report.txt", snippet="evidence")],
        )
    )

    response = make_client(service).post("/query", json={"query": "summarize reports"})

    assert response.status_code == 200
    service.query.assert_called_once_with("summarize reports", route=None)
    assert response.json()["route"] == "summary"


def test_query_endpoint_maps_unready_service() -> None:
    service = MagicMock()
    service.query = AsyncMock(side_effect=RuntimeError("The indexes are not ready"))

    response = make_client(service).post("/query", json={"query": "question"})

    assert response.status_code == 409
    assert response.json() == {"detail": "The indexes are not ready"}
