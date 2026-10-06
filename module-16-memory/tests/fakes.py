from __future__ import annotations

import math
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import UUID

from memory_system.models import (
    ActionEpisode,
    ChatMessage,
    LLMResponseModel,
    LongTermMemoryRecord,
    MemorySearchResult,
    ToolChoice,
    ToolSpec,
)


class CharacterCounter:
    # Counts characters as a predictable stand-in for a tokenizer.
    def count(self, text: str) -> int:
        return len(text)


class FixedEmbedder:
    # Returns a fixed vector for deterministic memory tests.
    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


class FakeLLMProvider:
    # Stores the response text that the fake provider should return.
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    # Returns a deterministic LLM response for extraction tests.
    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> LLMResponseModel:
        return LLMResponseModel(text=self.response_text, model="test", latency_ms=0)

    # Supplies an empty text stream to satisfy the provider protocol.
    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        if False:
            yield ""


class MemoryFakeRepository:
    # Initializes in-memory tables used by memory service tests.
    def __init__(self) -> None:
        self.memories: dict[UUID, LongTermMemoryRecord] = {}
        self.embeddings: dict[UUID, list[float]] = {}
        self.episodes: list[tuple[ActionEpisode, str]] = []

    # Saves a memory and its embedding in the in-memory store.
    async def save_memory(
        self, memory: LongTermMemoryRecord, embedding: list[float]
    ) -> LongTermMemoryRecord:
        self.memories[memory.id] = memory
        self.embeddings[memory.id] = embedding
        return memory

    # Returns the newest memories owned by the requested user.
    async def list_memories(self, user_id: str, limit: int) -> list[LongTermMemoryRecord]:
        records = [memory for memory in self.memories.values() if memory.user_id == user_id]
        return sorted(records, key=lambda record: record.created_at, reverse=True)[:limit]

    # Ranks a user's active memories by semantic match, recency, and supersession.
    async def search_memories(
        self, user_id: str, embedding: list[float], limit: int, half_life_days: int
    ) -> list[MemorySearchResult]:
        now = datetime.now(UTC)
        active = [
            memory
            for memory in self.memories.values()
            if memory.user_id == user_id and (memory.expires_at is None or memory.expires_at > now)
        ]
        superseded_ids = {memory_id for record in active for memory_id in record.supersedes_ids}
        results = []
        for memory in active:
            stored = self.embeddings[memory.id]
            similarity = sum(a * b for a, b in zip(embedding, stored, strict=True))
            age_days = max(0.0, (now - memory.created_at).total_seconds() / 86_400)
            recency = math.exp(-math.log(2) * age_days / half_life_days)
            superseded = memory.id in superseded_ids
            results.append(
                MemorySearchResult(
                    memory=memory,
                    similarity=similarity,
                    rank_score=0.65 * similarity + 0.35 * recency - (2.0 if superseded else 0),
                    superseded=superseded,
                )
            )
        return sorted(results, key=lambda result: result.rank_score, reverse=True)[:limit]

    # Deletes a memory only when it belongs to the specified user.
    async def delete_memory(self, user_id: str, memory_id: UUID) -> bool:
        memory = self.memories.get(memory_id)
        if memory is None or memory.user_id != user_id:
            return False
        del self.memories[memory_id]
        del self.embeddings[memory_id]
        return True

    # Removes all persistent memory and episode records for a user.
    async def delete_all_user_data(self, user_id: str) -> tuple[int, int]:
        memory_ids = [key for key, value in self.memories.items() if value.user_id == user_id]
        episode_ids = [
            index for index, (episode, _) in enumerate(self.episodes) if episode.user_id == user_id
        ]
        for memory_id in memory_ids:
            del self.memories[memory_id]
            del self.embeddings[memory_id]
        for index in reversed(episode_ids):
            del self.episodes[index]
        return len(memory_ids), len(episode_ids)

    # Adds an episode and its action fingerprint to the fake store.
    async def record_episode(self, episode: ActionEpisode, fingerprint: str) -> ActionEpisode:
        self.episodes.append((episode, fingerprint))
        return episode

    # Finds the newest unexpired episode matching the requested action.
    async def find_episode(
        self, user_id: str, task_id: str, fingerprint: str
    ) -> ActionEpisode | None:
        now = datetime.now(UTC)
        matches = [
            episode
            for episode, stored_fingerprint in self.episodes
            if episode.user_id == user_id
            and episode.task_id == task_id
            and stored_fingerprint == fingerprint
            and episode.expires_at > now
        ]
        return max(matches, key=lambda episode: episode.occurred_at) if matches else None

    # Deletes all episodes belonging to a user and returns the count removed.
    async def delete_episodes(self, user_id: str) -> int:
        before = len(self.episodes)
        self.episodes = [pair for pair in self.episodes if pair[0].user_id != user_id]
        return before - len(self.episodes)

    # Physically removes expired records and returns each layer's deletion count.
    async def purge_expired(self) -> tuple[int, int]:
        now = datetime.now(UTC)
        expired_memory_ids = [
            key
            for key, memory in self.memories.items()
            if memory.expires_at is not None and memory.expires_at <= now
        ]
        expired_episode_ids = [
            index for index, (episode, _) in enumerate(self.episodes) if episode.expires_at <= now
        ]
        for memory_id in expired_memory_ids:
            del self.memories[memory_id]
            del self.embeddings[memory_id]
        for index in reversed(expired_episode_ids):
            del self.episodes[index]
        return len(expired_memory_ids), len(expired_episode_ids)
