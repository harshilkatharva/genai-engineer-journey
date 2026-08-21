from uuid import uuid4

import pytest
from pydantic import ValidationError

from rag_app.models.retrive.retrive_response import (
    RetriveResponse,
    RetriveResult,
)


def test_retrive_result_with_valid_data():
    result = RetriveResult(
        chunk_text="Customers can request a refund within 30 days.",
        similarity_score=0.92,
    )

    assert result.chunk_text == "Customers can request a refund within 30 days."
    assert result.similarity_score == 0.92


def test_retrive_result_accepts_zero_similarity_score():
    result = RetriveResult(
        chunk_text="Some chunk",
        similarity_score=0.0,
    )

    assert result.similarity_score == 0.0


def test_retrive_result_accepts_negative_similarity_score():
    result = RetriveResult(
        chunk_text="Some chunk",
        similarity_score=-0.5,
    )

    assert result.similarity_score == -0.5


def test_retrive_result_requires_chunk_text():
    with pytest.raises(ValidationError):
        RetriveResult(
            similarity_score=0.92,
        )


def test_retrive_result_requires_similarity_score():
    with pytest.raises(ValidationError):
        RetriveResult(
            chunk_text="Some chunk",
        )


def test_retrive_response_with_valid_data():
    tenant_id = uuid4()

    results = [
        RetriveResult(
            chunk_text="Refunds are available within 30 days.",
            similarity_score=0.95,
        ),
        RetriveResult(
            chunk_text="Refund requests require an invoice.",
            similarity_score=0.87,
        ),
    ]

    response = RetriveResponse(
        tenant_id=tenant_id,
        queries=["What is the refund policy?"],
        results=results,
    )

    assert response.tenant_id == tenant_id
    assert response.queries == ["What is the refund policy?"]
    assert response.results == results


def test_retrive_response_accepts_multiple_queries():
    tenant_id = uuid4()

    response = RetriveResponse(
        tenant_id=tenant_id,
        queries=[
            "What is the refund policy?",
            "What is the cancellation policy?",
        ],
        results=[],
    )

    assert len(response.queries) == 2


def test_retrive_response_accepts_empty_results():
    response = RetriveResponse(
        tenant_id=uuid4(),
        queries=["test query"],
        results=[],
    )

    assert response.results == []


def test_retrive_response_requires_tenant_id():
    with pytest.raises(ValidationError):
        RetriveResponse(
            queries=["test query"],
            results=[],
        )


def test_retrive_response_rejects_invalid_tenant_id():
    with pytest.raises(ValidationError):
        RetriveResponse(
            tenant_id="not-a-uuid",
            queries=["test query"],
            results=[],
        )


def test_retrive_response_requires_queries():
    with pytest.raises(ValidationError):
        RetriveResponse(
            tenant_id=uuid4(),
            results=[],
        )


def test_retrive_response_requires_results():
    with pytest.raises(ValidationError):
        RetriveResponse(
            tenant_id=uuid4(),
            queries=["test query"],
        )


def test_retrive_response_builds_nested_retrive_result():
    response = RetriveResponse(
        tenant_id=uuid4(),
        queries=["What is the refund policy?"],
        results=[
            {
                "chunk_text": "Refunds are available within 30 days.",
                "similarity_score": 0.91,
            }
        ],
    )

    assert len(response.results) == 1
    assert isinstance(response.results[0], RetriveResult)
    assert response.results[0].chunk_text == ("Refunds are available within 30 days.")
    assert response.results[0].similarity_score == 0.91
