import pytest
from pydantic import ValidationError

from semantic_search_eng.models.process_request import ProcessRequest


def test_process_request_creation() -> None:
    request = ProcessRequest(
        tenant_id="conversation_123",
        documents=["document one", "document two"],
    )

    assert request.tenant_id == "conversation_123"
    assert request.documents == [
        "document one",
        "document two",
    ]


def test_process_request_rejects_empty_tenant_id() -> None:
    with pytest.raises(ValidationError):
        ProcessRequest(
            tenant_id="",
            documents=["document"],
        )


def test_process_request_rejects_empty_documents() -> None:
    with pytest.raises(ValidationError):
        ProcessRequest(
            tenant_id="conversation_123",
            documents=[],
        )
