from __future__ import annotations

import pytest

from memory_system import (
    EpisodicMemory,
    LongTermMemory,
    MemoryCandidate,
    MemoryService,
    WorkingMemoryRegistry,
)
from memory_system.core import Settings
from tests.fakes import FixedEmbedder, MemoryFakeRepository


# Verifies deletion removes memory from retrieval, storage, and user working context.
@pytest.mark.asyncio
async def test_delete_one_and_delete_all_remove_all_user_data() -> None:
    repository = MemoryFakeRepository()
    settings = Settings(memory_embedding_dimension=2)
    long_term = LongTermMemory(repository, FixedEmbedder(), settings=settings)
    episodic = EpisodicMemory(repository, settings)
    working = WorkingMemoryRegistry()
    active = working.get("user-1", "conversation-1")
    active.pin_goal("Keep this private goal")
    service = MemoryService(long_term, episodic, working)
    first = await long_term.store("user-1", MemoryCandidate(content="First memory"))
    await long_term.store("user-1", MemoryCandidate(content="Second memory"))
    await episodic.record("user-1", "task-1", "update", "record", status="attempted")

    assert await service.delete_memory("user-1", first.id) is True
    assert all(
        result.memory.id != first.id
        for result in await long_term.retrieve("user-1", "First memory")
    )
    assert len(await service.view_memories("user-1")) == 1

    deleted = await service.delete_all_memories("user-1")

    assert deleted == {"long_term": 1, "episodic": 1}
    assert await service.view_memories("user-1") == []
    assert await long_term.retrieve("user-1", "Second memory") == []
    check = await episodic.check("user-1", "task-1", "update", "record")
    assert check.already_attempted is False
    assert working.get("user-1", "conversation-1") is not active
    assert repository.memories == {}
    assert repository.episodes == []
