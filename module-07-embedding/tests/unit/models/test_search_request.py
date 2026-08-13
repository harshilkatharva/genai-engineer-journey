import pytest
from pydantic import ValidationError

from semantic_search_eng.models.search_request import SearchRequest


def test_search_request_defaults_top_k() -> None:
    request = SearchRequest(
        conversation_id="conversation_123",
        query="How does authentication work?",
    )

    assert request.top_k == 5


def test_search_request_creation() -> None:
    request = SearchRequest(
        conversation_id="conversation_123",
        query="How does authentication work?",
        top_k=10,
    )

    assert request.conversation_id == "conversation_123"
    assert request.query == "How does authentication work?"
    assert request.top_k == 10


@pytest.mark.parametrize(
    "field, value",
    [
        ("conversation_id", ""),
        ("query", ""),
    ],
)
def test_search_request_rejects_empty_strings(
    field: str,
    value: str,
) -> None:
    data = {
        "conversation_id": "conversation_123",
        "query": "test query",
    }

    data[field] = value

    with pytest.raises(ValidationError):
        SearchRequest(**data)


def test_search_request_rejects_invalid_top_k() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(
            conversation_id="conversation_123",
            query="test",
            top_k=0,
        )


def test_search_request_rejects_top_k_above_limit() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(
            conversation_id="conversation_123",
            query="test",
            top_k=101,
        )
