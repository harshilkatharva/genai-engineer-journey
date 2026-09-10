from unittest.mock import AsyncMock
from uuid import uuid4

from fastapi.testclient import TestClient

from rag_app.api.app import app
from rag_app.api.routes import rag as rag_route
from rag_app.models import (
    RAGClassificationResponse,
    RAGExtractionResponse,
    RAGRequest,
    RAGResposne,
)


client = TestClient(app)


def test_chat_answer_endpoint_uses_chat_feature(monkeypatch):
    tenant_id = uuid4()
    expected = RAGResposne(text="Grounded answer", source="policy.pdf")
    get_chat_answer = AsyncMock(return_value=expected)
    monkeypatch.setattr(rag_route.rag_chat, "get_chat_answer", get_chat_answer)

    response = client.post(
        "/rag/chat_answer",
        json={"tenant_id": str(tenant_id), "query": "What is the policy?"},
    )

    assert response.status_code == 200
    assert response.json() == expected.model_dump()
    request = get_chat_answer.await_args.args[0]
    assert isinstance(request, RAGRequest)
    assert request.tenant_id == tenant_id
    assert request.query == "What is the policy?"


def test_chat_answer_endpoint_accepts_query_dict(monkeypatch):
    tenant_id = uuid4()
    expected = RAGResposne(text="Grounded answer", source=None)
    get_chat_answer = AsyncMock(return_value=expected)
    monkeypatch.setattr(rag_route.rag_chat, "get_chat_answer", get_chat_answer)

    response = client.post(
        "/rag/chat_answer",
        json={
            "tenant_id": str(tenant_id),
            "query": {"text": "What is the policy?"},
        },
    )

    assert response.status_code == 200
    assert get_chat_answer.await_args.args[0].query == "What is the policy?"


def test_classification_endpoint_uses_classification_feature(monkeypatch):
    expected = RAGClassificationResponse(
        category="billing",
        confidence=0.9,
        reason="Invoice text.",
    )
    classify = AsyncMock(return_value=expected)
    monkeypatch.setattr(rag_route.rag_classification, "classify", classify)

    response = client.post("/rag/classification", json={"text": "Invoice issue"})

    assert response.status_code == 200
    assert response.json() == expected.model_dump()
    classify.assert_awaited_once_with("Invoice issue")


def test_extraction_endpoint_uses_extraction_feature(monkeypatch):
    expected = RAGExtractionResponse(fields={"invoice_id": "INV-1"})
    extract = AsyncMock(return_value=expected)
    monkeypatch.setattr(rag_route.rag_extraction, "extract", extract)

    response = client.post("/rag/extraction", json={"text": "Invoice INV-1"})

    assert response.status_code == 200
    assert response.json() == expected.model_dump()
    extract.assert_awaited_once_with("Invoice INV-1")


def test_feature_endpoints_reject_empty_text():
    for path in ("/rag/classification", "/rag/extraction"):
        response = client.post(path, json={"text": ""})
        assert response.status_code == 422
