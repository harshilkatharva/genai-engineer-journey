from __future__ import annotations

from typing import Any

import pytest

from memory_system.models import MemorySearchResult
from memory_system.retrieval.manager import MemoryRetrievalManager
from tests.fakes import MemoryFakeRepository, make_settings


class FakeEmbedder:
    def __init__(self, vector: list[Any] | None = None) -> None:
        self.vector = vector or [0.5, 0.5]
        self.queries: list[str] = []

    async def embed(self, text: str) -> list[Any]:
        self.queries.append(text)
        return self.vector


@pytest.mark.asyncio
async def test_retrieve_embeds_query_and_passes_configured_defaults() -> None:
    class RecordingRepository(MemoryFakeRepository):
        async def search_memories(
            self, user_id: str, embedding: list[float], limit: int, half_life_days: int
        ) -> list[MemorySearchResult]:
            self.query = (user_id, embedding, limit, half_life_days)
            return []

    repository = RecordingRepository()
    embedder = FakeEmbedder()
    settings = make_settings(
        memory_embedding_dimension=2,
        memory_retrieval_limit=4,
        memory_recency_half_life_days=17,
    )
    manager = MemoryRetrievalManager(repository, embedder, settings)

    assert await manager.retrieve("user-1", "  favorite drink ") == []
    assert embedder.queries == ["  favorite drink "]
    assert repository.query == ("user-1", [0.5, 0.5], 4, 17)
    assert await manager.retrieve("user-1", "limit boundary", 500) == []
    assert repository.query == ("user-1", [0.5, 0.5], 500, 17)


@pytest.mark.asyncio
async def test_retrieve_rejects_missing_inputs_and_invalid_limits() -> None:
    manager = MemoryRetrievalManager(
        MemoryFakeRepository(), FakeEmbedder(), make_settings(memory_embedding_dimension=2)
    )

    invalid_requests = [
        ("", "hello", None),
        ("user-1", "  ", None),
        ("user-1", "hello", 0),
        ("user-1", "hello", 501),
        ("user-1", "hello", True),
    ]
    for user_id, query, limit in invalid_requests:
        with pytest.raises(ValueError):
            await manager.retrieve(user_id, query, limit)


@pytest.mark.asyncio
async def test_retrieve_rejects_invalid_embedding_vectors() -> None:
    invalid_vectors: list[tuple[list[Any], str]] = [
        ([1.0], "1 dimensions"),
        ([True, 0.0], "non-finite or non-numeric"),
        ([float("inf"), 0.0], "non-finite or non-numeric"),
        (["1", 0.0], "non-finite or non-numeric"),
    ]
    for vector, message in invalid_vectors:
        manager = MemoryRetrievalManager(
            MemoryFakeRepository(),
            FakeEmbedder(vector),
            make_settings(memory_embedding_dimension=2),
        )
        with pytest.raises(ValueError, match=message):
            await manager.retrieve("user-1", "hello")
