import pytest
from pydantic import ValidationError

from semantic_search_eng.models.request_model import RequestModel


def test_request_model_creation() -> None:
    request = RequestModel(
        tenant_id="conversation_123",
    )

    assert request.tenant_id == "conversation_123"


def test_request_model_rejects_empty_tenant_id() -> None:
    with pytest.raises(ValidationError):
        RequestModel(
            tenant_id="",
        )
