import pytest
from pydantic import ValidationError

from rag_app.models import RetriveRequest


def test_retrive_request_defaults_top_k() -> None:
    request = RetriveRequest(
        tenant_id="550e8400-e29b-41d4-a716-446655440001",
        queries=["How does authentication work?"],
    )

    assert request.top_k_candidate == 5
    assert request.top_k_re_ranker == 5


def test_retrive_request_creation() -> None:
    request = RetriveRequest(
        tenant_id="550e8400-e29b-41d4-a716-446655440001",
        queries=["How does authentication work?"],
        top_k_candidate=50,
        top_k_re_ranker=5,
    )

    assert request.tenant_id == "550e8400-e29b-41d4-a716-446655440001"
    assert request.queries == ["How does authentication work?"]
    assert request.top_k_candidate == 50
    assert request.top_k_re_ranker == 5
    assert request.document_type is None


def test_retrive_request_creation_with_document_type() -> None:
    request = RetriveRequest(
        tenant_id="550e8400-e29b-41d4-a716-446655440001",
        queries=["How does authentication work?"],
        document_type=["PDF"],
    )

    assert request.tenant_id == "550e8400-e29b-41d4-a716-446655440001"
    assert request.queries == ["How does authentication work?"]
    assert request.document_type == ["PDF"]


@pytest.mark.parametrize(
    "field, value",
    [
        ("tenant_id", ""),
        ("queries", ""),
    ],
)
def test_retrive_request_rejects_empty_strings(
    field: str,
    value: str,
) -> None:
    data = {
        "tenant_id": "550e8400-e29b-41d4-a716-446655440001",
        "queries": "test queries",
    }

    data[field] = value

    with pytest.raises(ValidationError):
        RetriveRequest(**data)


def test_retrive_request_rejects_invalid_top_k() -> None:
    with pytest.raises(ValidationError):
        RetriveRequest(
            tenant_id="conversation_123",
            queries=["test"],
            top_k_candidate=0,
        )


def test_search_request_rejects_top_k_above_limit() -> None:
    with pytest.raises(ValidationError):
        RetriveRequest(
            tenant_id="conversation_123",
            queries=["test"],
            top_k_re_ranker=101,
        )
