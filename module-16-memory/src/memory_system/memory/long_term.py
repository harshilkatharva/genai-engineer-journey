from __future__ import annotations

import math
from uuid import UUID

from ..core import Settings, get_settings
from ..db.repository import MemoryRepository
from ..embedding.provider import EmbeddingProvider, SentenceTransformerEmbeddings
from ..models import (
    ConversationMessage,
    LongTermMemoryRecord,
    MemoryCandidate,
    MemorySearchResult,
)
from ..retrieval.manager import MemoryRetrievalManager
from .extraction import MemoryExtractor, validate_memory_content


class LongTermMemory:
    # Combines persistence, embeddings, extraction, and semantic retrieval services.
    def __init__(
        self,
        repository: MemoryRepository,
        embedder: EmbeddingProvider | None = None,
        extractor: MemoryExtractor | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.repository = repository
        self.settings = settings or get_settings()
        self.embedder = embedder or SentenceTransformerEmbeddings(
            self.settings.memory_embedding_model,
            self.settings.memory_embedding_dimension,
        )
        self.extractor = extractor
        self.retrieval = MemoryRetrievalManager(repository, self.embedder, self.settings)

    # Embeds and persists a validated memory for the specified user.
    async def store(self, user_id: str, candidate: MemoryCandidate) -> LongTermMemoryRecord:
        if not user_id:
            raise ValueError("user_id is required")
        validate_memory_content(candidate.content)
        existing = await self.repository.list_memories(user_id, limit=500)
        existing_ids = {memory.id for memory in existing}
        unknown = set(candidate.supersedes_ids) - existing_ids
        if unknown:
            raise ValueError("supersedes_ids must reference this user's existing memories")
        record = LongTermMemoryRecord(
            user_id=user_id,
            content=candidate.content,
            category=candidate.category,
            supersedes_ids=candidate.supersedes_ids,
            expires_at=candidate.expires_at,
        )
        vector = await self._embed(record.content)
        return await self.repository.save_memory(record, vector)

    # Extracts durable memory candidates from messages and stores each one.
    async def remember_conversation(
        self, user_id: str, messages: list[ConversationMessage]
    ) -> list[LongTermMemoryRecord]:
        if self.extractor is None:
            raise RuntimeError("An LLM memory extractor is required for conversation extraction")
        existing = await self.repository.list_memories(user_id, limit=500)
        candidates = await self.extractor.extract(messages, existing)
        known_ids = {memory.id for memory in existing}
        if any(
            memory_id not in known_ids
            for candidate in candidates
            for memory_id in candidate.supersedes_ids
        ):
            raise ValueError(
                "Extracted supersedes_ids must reference this user's existing memories"
            )
        records: list[LongTermMemoryRecord] = []
        for candidate in candidates:
            records.append(await self.store(user_id, candidate))
        return records

    # Retrieves relevant user memories using semantic and recency ranking.
    async def retrieve(
        self, user_id: str, query: str, limit: int | None = None
    ) -> list[MemorySearchResult]:
        return await self.retrieval.retrieve(user_id, query, limit)

    # Lists a user's active long-term memories for review.
    async def view(self, user_id: str, limit: int = 100) -> list[LongTermMemoryRecord]:
        if not user_id:
            raise ValueError("user_id is required")
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        return await self.repository.list_memories(user_id, limit)

    # Deletes a specific long-term memory owned by the user.
    async def delete(self, user_id: str, memory_id: UUID) -> bool:
        if not user_id:
            raise ValueError("user_id is required")
        return await self.repository.delete_memory(user_id, memory_id)

    # Embeds text and validates that the returned vector matches configuration.
    async def _embed(self, text: str) -> list[float]:
        vector = await self.embedder.embed(text)
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
        return [float(value) for value in vector]
