from __future__ import annotations

from uuid import UUID

from ..memory.episodic import EpisodicMemory
from ..memory.long_term import LongTermMemory
from ..memory.working import WorkingMemoryRegistry
from ..models import LongTermMemoryRecord


class MemoryService:
    # Connects memory layers and user-facing management operations.
    def __init__(
        self,
        long_term: LongTermMemory,
        episodic: EpisodicMemory,
        working: WorkingMemoryRegistry | None = None,
    ) -> None:
        if long_term.repository is not episodic.repository:
            raise ValueError("Long-term and episodic memory must share a deletion-capable store")
        self.long_term = long_term
        self.episodic = episodic
        self.working = working or WorkingMemoryRegistry()

    # Returns the user's visible long-term memories.
    async def view_memories(self, user_id: str, limit: int = 100) -> list[LongTermMemoryRecord]:
        return await self.long_term.view(user_id, limit)

    # Deletes one long-term memory belonging to the user.
    async def delete_memory(self, user_id: str, memory_id: UUID) -> bool:
        return await self.long_term.delete(user_id, memory_id)

    # Deletes all persistent user memories and clears local working context.
    async def delete_all_memories(self, user_id: str) -> dict[str, int]:
        if not user_id:
            raise ValueError("user_id is required")
        self.working.clear_user(user_id)
        memory_count, episode_count = await self.long_term.repository.delete_all_user_data(user_id)
        return {"long_term": memory_count, "episodic": episode_count}

    # Physically removes expired persistent memories and returns deletion counts.
    async def purge_expired(self) -> dict[str, int]:
        memory_count, episode_count = await self.long_term.repository.purge_expired()
        return {"long_term": memory_count, "episodic": episode_count}


__all__ = ["MemoryService"]
