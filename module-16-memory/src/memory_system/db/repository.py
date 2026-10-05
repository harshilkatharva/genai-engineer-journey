from __future__ import annotations

from typing import Protocol
from uuid import UUID

from ..models import ActionEpisode, LongTermMemoryRecord, MemorySearchResult


class MemoryRepository(Protocol):
    async def save_memory(
        self, memory: LongTermMemoryRecord, embedding: list[float]
    ) -> LongTermMemoryRecord: ...

    async def list_memories(self, user_id: str, limit: int) -> list[LongTermMemoryRecord]: ...

    async def search_memories(
        self, user_id: str, embedding: list[float], limit: int, half_life_days: int
    ) -> list[MemorySearchResult]: ...

    async def delete_memory(self, user_id: str, memory_id: UUID) -> bool: ...

    async def delete_all_user_data(self, user_id: str) -> tuple[int, int]: ...

    async def purge_expired(self) -> tuple[int, int]: ...

    async def record_episode(self, episode: ActionEpisode, fingerprint: str) -> ActionEpisode: ...

    async def find_episode(
        self, user_id: str, task_id: str, fingerprint: str
    ) -> ActionEpisode | None: ...

    async def delete_episodes(self, user_id: str) -> int: ...
