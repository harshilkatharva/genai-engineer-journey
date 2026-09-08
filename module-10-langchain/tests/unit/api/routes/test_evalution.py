from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from rag_app.api.app import app
from rag_app.api.routes import evalution as evalution_route


client = TestClient(app)


def test_report_endpoint_uses_report_service(monkeypatch):
    get_report = MagicMock(return_value={"accuracy": 0.9})
    monkeypatch.setattr(evalution_route.evalution_report, "get_report", get_report)

    response = client.get("/evalution/report")

    assert response.status_code == 200
    assert response.json() == {"accuracy": 0.9}
    get_report.assert_called_once_with()


def test_test_queries_endpoint_uses_query_runner(monkeypatch):
    call_queries = MagicMock(return_value={"status": "completed"})
    monkeypatch.setattr(evalution_route, "call_queries", call_queries)

    response = client.get("/evalution/test_queries")

    assert response.status_code == 200
    assert response.json() == {"status": "completed"}
    call_queries.assert_called_once_with()
