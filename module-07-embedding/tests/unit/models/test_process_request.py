import pytest
from pydantic import ValidationError

from semantic_search_eng.models.process_request import ProcessRequest


def test_process_request_creation() -> None:
    request = ProcessRequest(
        conversation_id="conversation_123",
        documents=["document one", "document two"],
    )

    assert request.conversation_id == "conversation_123"
    assert request.documents == [
        "document one",
        "document two",
    ]


def test_process_request_rejects_empty_conversation_id() -> None:
    with pytest.raises(ValidationError):
        ProcessRequest(
            conversation_id="",
            documents=["document"],
        )


def test_process_request_rejects_empty_documents() -> None:
    with pytest.raises(ValidationError):
        ProcessRequest(
            conversation_id="conversation_123",
            documents=[],
        )
