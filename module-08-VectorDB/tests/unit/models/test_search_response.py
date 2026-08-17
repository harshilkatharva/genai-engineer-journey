from semantic_search_eng.models.chunk import Chunk
from semantic_search_eng.models.retrive_response import (
    SearchResponse,
    SearchResult,
)


def create_chunk() -> Chunk:
    return Chunk(
        chunk_id="document_0000_chunk_0001",
        document_id="document_0000",
        tenant_id="conversation_123",
        chunk_index=1,
        text="Authentication information.",
        token_count=3,
    )


def test_search_result_creation() -> None:
    result = SearchResult(
        chunk=create_chunk(),
        similarity_score=0.87,
    )

    assert result.chunk.chunk_id == ("document_0000_chunk_0001")
    assert result.similarity_score == 0.87


def test_search_response_creation() -> None:
    result = SearchResult(
        chunk=create_chunk(),
        similarity_score=0.87,
    )

    response = SearchResponse(
        tenant_id="conversation_123",
        query="How does authentication work?",
        top_k=5,
        results=[result],
    )

    assert response.tenant_id == "conversation_123"
    assert response.query == ("How does authentication work?")
    assert response.top_k == 5
    assert len(response.results) == 1
