from rag_app.models import (
    RetriveResponse,
    RetriveResult,
)


def test_retrive_result_creation() -> None:
    result = RetriveResult(
        chunk_text="Authentication information.",
        similarity_score=0.87,
    )

    assert result.chunk_text == "Authentication information."
    assert result.similarity_score == 0.87


def test_retrive_response_creation() -> None:
    result = RetriveResult(
        chunk_text="Authentication information.",
        similarity_score=0.87,
    )

    response = RetriveResponse(
        tenant_id="550e8400-e29b-41d4-a716-446655440001",
        query="How does authentication work?",
        top_k=5,
        results=[result],
    )

    assert response.tenant_id == "550e8400-e29b-41d4-a716-446655440001"
    assert response.query == "How does authentication work?"
    assert response.top_k == 5
    assert len(response.results) == 1
