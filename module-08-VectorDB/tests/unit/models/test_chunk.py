import pytest
from pydantic import ValidationError

from semantic_search_eng.models.chunk import Chunk


def test_chunk_creation() -> None:
    chunk = Chunk(
        chunk_id="document_0000_chunk_0001",
        document_id="document_0000",
        tenant_id="conversation_123",
        chunk_index=1,
        text="This is a test chunk.",
        token_count=5,
        start_position=10,
        end_position=31,
    )

    assert chunk.chunk_id == "document_0000_chunk_0001"
    assert chunk.document_id == "document_0000"
    assert chunk.tenant_id == "conversation_123"
    assert chunk.chunk_index == 1
    assert chunk.text == "This is a test chunk."
    assert chunk.token_count == 5
    assert chunk.start_position == 10
    assert chunk.end_position == 31


def test_chunk_allows_optional_positions() -> None:
    chunk = Chunk(
        chunk_id="chunk_1",
        document_id="doc_1",
        tenant_id="conversation_1",
        chunk_index=0,
        text="Test",
        token_count=1,
    )

    assert chunk.start_position is None
    assert chunk.end_position is None


def test_chunk_rejects_negative_index() -> None:
    with pytest.raises(ValidationError):
        Chunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            tenant_id="conversation_1",
            chunk_index=-1,
            text="Test",
            token_count=1,
        )


def test_chunk_rejects_negative_token_count() -> None:
    with pytest.raises(ValidationError):
        Chunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            tenant_id="conversation_1",
            chunk_index=0,
            text="Test",
            token_count=-1,
        )
