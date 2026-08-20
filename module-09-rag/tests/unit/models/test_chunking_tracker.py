from rag_app.models import (
    ChunkingTracker,
)


def test_chunking_tracker_creation() -> None:
    tracker = ChunkingTracker(
        tenant_id="conversation_123",
        document_count=2,
        total_chunks=20,
        chunking_strategy="sentence",
        chunk_size=500,
        overlap=50,
        total_input_tokens=9000,
        latency_ms=125.5,
    )

    assert tracker.tenant_id == "conversation_123"
    assert tracker.document_count == 2
    assert tracker.total_chunks == 20
    assert tracker.chunking_strategy == "sentence"
    assert tracker.chunk_size == 500
    assert tracker.overlap == 50
    assert tracker.total_input_tokens == 9000
    assert tracker.latency_ms == 125.5
