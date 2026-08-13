from semantic_search_eng.models.query_tracker import QueryTracker


def test_query_tracker_creation() -> None:
    tracker = QueryTracker(
        conversation_id="conversation_123",
        query="How does authentication work?",
        embedding_model="all-MiniLM-L6-v2",
        query_token_count=5,
        embedding_latency_ms=12.5,
        retrieval_latency_ms=2.5,
        total_latency_ms=15.0,
        estimated_cost=0.0,
        top_k=5,
    )

    assert tracker.conversation_id == "conversation_123"
    assert tracker.query == ("How does authentication work?")
    assert tracker.embedding_model == "all-MiniLM-L6-v2"
    assert tracker.query_token_count == 5
    assert tracker.embedding_latency_ms == 12.5
    assert tracker.retrieval_latency_ms == 2.5
    assert tracker.total_latency_ms == 15.0
    assert tracker.estimated_cost == 0.0
    assert tracker.top_k == 5
