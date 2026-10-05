from dataclasses import dataclass

import numpy as np
import pytest


@dataclass
class UserMemory:
    id: int
    user_id: str
    content: str
    embedding: np.ndarray


class UserMemoryStore:
    def __init__(self):
        self.memories: list[UserMemory] = []

    def add(self, memory: UserMemory):
        self.memories.append(memory)

    # --------------------------------------------------------
    # Delete Memory
    # --------------------------------------------------------

    def delete_user_memory(
        self,
        memory_id: int,
        user_id: str,
    ) -> bool:
        for memory in self.memories:
            if memory.id == memory_id and memory.user_id == user_id:
                self.memories.remove(memory)
                return True

        return False

    def retrieve(
        self,
        user_id: str,
        query_embedding: np.ndarray,
        limit: int = 5,
    ) -> list[UserMemory]:
        user_memories = [memory for memory in self.memories if memory.user_id == user_id]

        # Mock similarity search
        def similarity(memory):
            return np.dot(
                query_embedding,
                memory.embedding,
            )

        return sorted(
            user_memories,
            key=similarity,
            reverse=True,
        )[:limit]


@pytest.fixture
def memory_store():
    store = UserMemoryStore()

    store.add(
        UserMemory(
            id=1,
            user_id="user-1",
            content="User prefers Python.",
            embedding=np.array([1.0, 0.0, 0.0]),
        )
    )

    store.add(
        UserMemory(
            id=2,
            user_id="user-1",
            content="User is learning FastAPI.",
            embedding=np.array([0.9, 0.1, 0.0]),
        )
    )

    store.add(
        UserMemory(
            id=3,
            user_id="user-1",
            content="User likes PostgreSQL.",
            embedding=np.array([0.8, 0.2, 0.0]),
        )
    )

    return store


def test_deleted_memory_never_appears_in_retrieval(
    memory_store,
):
    query_embedding = np.array([1.0, 0.0, 0.0])

    # Verify memory exists before deletion
    before_delete = memory_store.retrieve(
        user_id="user-1",
        query_embedding=query_embedding,
    )

    assert any(memory.id == 1 for memory in before_delete)

    # Delete memory
    deleted = memory_store.delete_user_memory(
        memory_id=1,
        user_id="user-1",
    )

    assert deleted is True

    # Retrieve again
    after_delete = memory_store.retrieve(
        user_id="user-1",
        query_embedding=query_embedding,
    )

    # Deleted memory must never appear
    assert not any(memory.id == 1 for memory in after_delete)


def test_user_cannot_delete_another_users_memory(
    memory_store,
):
    deleted = memory_store.delete_user_memory(
        memory_id=1,
        user_id="user-2",
    )

    assert deleted is False

    memories = memory_store.retrieve(
        user_id="user-1",
        query_embedding=np.array([1.0, 0.0, 0.0]),
    )

    assert any(memory.id == 1 for memory in memories)


def test_delete_non_existing_memory(
    memory_store,
):
    deleted = memory_store.delete_user_memory(
        memory_id=999,
        user_id="user-1",
    )

    assert deleted is False
