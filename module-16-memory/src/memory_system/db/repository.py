from __future__ import annotations

from typing import Protocol
from uuid import UUID

from ..models import ActionEpisode, LongTermMemoryRecord, MemorySearchResult


class MemoryRepository(Protocol):
    # Stores a long-term memory together with its vector embedding.
    async def save_memory(
        self, memory: LongTermMemoryRecord, embedding: list[float]
    ) -> LongTermMemoryRecord: ...

    # Lists a user's active long-term memories.
    async def list_memories(self, user_id: str, limit: int) -> list[LongTermMemoryRecord]: ...

    # Searches a user's memories using semantic similarity and recency.
    async def search_memories(
        self, user_id: str, embedding: list[float], limit: int, half_life_days: int
    ) -> list[MemorySearchResult]: ...

    # Deletes a specific memory belonging to a user.
    async def delete_memory(self, user_id: str, memory_id: UUID) -> bool: ...

    # Deletes all persistent long-term and episodic data for a user.
    async def delete_all_user_data(self, user_id: str) -> tuple[int, int]: ...

    # Removes all persistent memory records that have expired.
    async def purge_expired(self) -> tuple[int, int]: ...

    # Stores a structured record of an attempted action.
    async def record_episode(self, episode: ActionEpisode, fingerprint: str) -> ActionEpisode: ...

    # Looks up an unexpired episode matching a user's task and action.
    async def find_episode(
        self, user_id: str, task_id: str, fingerprint: str
    ) -> ActionEpisode | None: ...

    # Deletes all episodic records for a user.
    async def delete_episodes(self, user_id: str) -> int: ...
