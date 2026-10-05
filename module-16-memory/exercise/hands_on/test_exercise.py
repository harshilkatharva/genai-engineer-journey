import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pytest

# task 4

RECENCY_HALF_LIFE_DAYS = 30


@dataclass
class Memory:
    content: str
    embedding: np.ndarray
    created_at: datetime


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def score_memory(
    memory: Memory,
    query_embedding: np.ndarray,
    now: datetime,
) -> float:
    similarity = cosine_similarity(
        query_embedding,
        memory.embedding,
    )

    days_old = (now - memory.created_at).days

    recency_score = math.exp(-days_old / RECENCY_HALF_LIFE_DAYS)

    return 0.7 * similarity + 0.3 * recency_score


def similarity_retrieval(
    memories: list[Memory],
    query_embedding: np.ndarray,
    top_k: int = 3,
) -> list[Memory]:
    return sorted(
        memories,
        key=lambda memory: cosine_similarity(
            query_embedding,
            memory.embedding,
        ),
        reverse=True,
    )[:top_k]


def relevance_recency_retrieval(
    memories: list[Memory],
    query_embedding: np.ndarray,
    now: datetime,
    top_k: int = 3,
) -> list[Memory]:
    return sorted(
        memories,
        key=lambda memory: score_memory(
            memory,
            query_embedding,
            now,
        ),
        reverse=True,
    )[:top_k]


@pytest.fixture
def test_data():
    now = datetime(2026, 10, 5)

    query_embedding = np.array([1.0, 0.0, 0.0])

    memories = [
        # Old memory:
        # Highest semantic similarity
        Memory(
            content="User prefers Python for backend development.",
            embedding=np.array([0.99, 0.01, 0.0]),
            created_at=now - timedelta(days=365),
        ),
        # Recent memory:
        # Slightly lower similarity but much more recent
        Memory(
            content=("User recently started preferring FastAPI for backend projects."),
            embedding=np.array([0.90, 0.10, 0.0]),
            created_at=now - timedelta(days=5),
        ),
        # Less relevant memory
        Memory(
            content="User is learning React.",
            embedding=np.array([0.30, 0.70, 0.0]),
            created_at=now - timedelta(days=10),
        ),
    ]

    return memories, query_embedding, now


def test_pure_similarity_returns_old_memory_first(
    test_data,
):
    memories, query_embedding, now = test_data

    results = similarity_retrieval(
        memories,
        query_embedding,
    )

    assert results[0].content == "User prefers Python for backend development."


def test_relevance_recency_prioritizes_recent_memory(
    test_data,
):
    memories, query_embedding, now = test_data

    results = relevance_recency_retrieval(
        memories,
        query_embedding,
        now,
    )

    assert results[0].content == ("User recently started preferring FastAPI for backend projects.")


def test_old_memory_is_demoted(
    test_data,
):
    memories, query_embedding, now = test_data

    similarity_results = similarity_retrieval(
        memories,
        query_embedding,
    )

    recency_results = relevance_recency_retrieval(
        memories,
        query_embedding,
        now,
    )

    old_memory = "User prefers Python for backend development."

    recent_memory = "User recently started preferring FastAPI for backend projects."

    # Pure similarity -> old memory wins
    assert similarity_results[0].content == old_memory

    # Relevance + recency -> recent memory wins
    assert recency_results[0].content == recent_memory


def test_recency_score_is_between_zero_and_one(
    test_data,
):
    memories, query_embedding, now = test_data

    for memory in memories:
        days_old = (now - memory.created_at).days

        recency_score = math.exp(-days_old / RECENCY_HALF_LIFE_DAYS)

        assert 0.0 < recency_score <= 1.0


if __name__ == "__main__":
    now = datetime(2026, 10, 5)

    query_embedding = np.array([1.0, 0.0, 0.0])

    memories = [
        Memory(
            content="User prefers Python for backend development.",
            embedding=np.array([0.99, 0.01, 0.0]),
            created_at=now - timedelta(days=365),
        ),
        Memory(
            content=("User recently started preferring FastAPI for backend projects."),
            embedding=np.array([0.90, 0.10, 0.0]),
            created_at=now - timedelta(days=5),
        ),
        Memory(
            content="User is learning React.",
            embedding=np.array([0.30, 0.70, 0.0]),
            created_at=now - timedelta(days=10),
        ),
    ]

    print("=" * 60)
    print("PURE SIMILARITY RETRIEVAL")
    print("=" * 60)

    results = similarity_retrieval(
        memories,
        query_embedding,
    )

    for memory in results:
        similarity = cosine_similarity(
            query_embedding,
            memory.embedding,
        )

        print(f"{similarity:.3f} -> {memory.content}")

    print()
    print("=" * 60)
    print("RELEVANCE + RECENCY RETRIEVAL")
    print("=" * 60)

    results = relevance_recency_retrieval(
        memories,
        query_embedding,
        now,
    )

    for memory in results:
        score = score_memory(
            memory,
            query_embedding,
            now,
        )

        print(f"{score:.3f} -> {memory.content}")
