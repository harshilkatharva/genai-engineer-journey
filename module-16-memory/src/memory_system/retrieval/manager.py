from __future__ import annotations

import math

from ..core import Settings, get_settings
from ..db.repository import MemoryRepository
from ..embedding.provider import EmbeddingProvider, SentenceTransformerEmbeddings
from ..models import MemorySearchResult


class MemoryRetrievalManager:
    def __init__(
        self,
        repository: MemoryRepository,
        embedder: EmbeddingProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.repository = repository
        self.settings = settings or get_settings()
        self.embedder = embedder or SentenceTransformerEmbeddings(
            self.settings.memory_embedding_model,
            self.settings.memory_embedding_dimension,
        )

    async def retrieve(
        self, user_id: str, query: str, limit: int | None = None
    ) -> list[MemorySearchResult]:
        if not user_id or not query.strip():
            raise ValueError("user_id and a non-empty query are required")
        result_limit = self.settings.memory_retrieval_limit if limit is None else limit
        if isinstance(result_limit, bool) or not 1 <= result_limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        vector = await self.embedder.embed(query)
        if len(vector) != self.settings.memory_embedding_dimension:
            raise ValueError(
                f"Embedding provider returned {len(vector)} dimensions; configured for "
                f"{self.settings.memory_embedding_dimension}"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in vector
        ):
            raise ValueError("Embedding provider returned a non-finite or non-numeric vector")
        return await self.repository.search_memories(
            user_id,
            vector,
            result_limit,
            self.settings.memory_recency_half_life_days,
        )
