import pytest
from pydantic import ValidationError

from semantic_search_eng.models.request_model import RequestModel


def test_request_model_creation() -> None:
    request = RequestModel(
        conversation_id="conversation_123",
    )

    assert request.conversation_id == "conversation_123"


def test_request_model_rejects_empty_conversation_id() -> None:
    with pytest.raises(ValidationError):
        RequestModel(
            conversation_id="",
        )
