import pytest
from pydantic import ValidationError

from semantic_search_eng.models import RetriveRequest


def test_retrive_request_defaults_top_k() -> None:
    request = RetriveRequest(
        tenant_id="550e8400-e29b-41d4-a716-446655440001",
        query="How does authentication work?",
    )

    assert request.top_k == 5


def test_retrive_request_creation() -> None:
    request = RetriveRequest(
        tenant_id="550e8400-e29b-41d4-a716-446655440001",
        query="How does authentication work?",
        top_k=10,
    )

    assert request.tenant_id == "550e8400-e29b-41d4-a716-446655440001"
    assert request.query == "How does authentication work?"
    assert request.top_k == 10


@pytest.mark.parametrize(
    "field, value",
    [
        ("tenant_id", ""),
        ("query", ""),
    ],
)
def test_retrive_request_rejects_empty_strings(
    field: str,
    value: str,
) -> None:
    data = {
        "tenant_id": "550e8400-e29b-41d4-a716-446655440001",
        "query": "test query",
    }

    data[field] = value

    with pytest.raises(ValidationError):
        RetriveRequest(**data)


def test_retrive_request_rejects_invalid_top_k() -> None:
    with pytest.raises(ValidationError):
        RetriveRequest(
            tenant_id="conversation_123",
            query="test",
            top_k=0,
        )


def test_search_request_rejects_top_k_above_limit() -> None:
    with pytest.raises(ValidationError):
        RetriveRequest(
            tenant_id="conversation_123",
            query="test",
            top_k=101,
        )
