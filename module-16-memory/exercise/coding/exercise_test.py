def fake_memory_store():
    class FakeMemoryStore:
        def __init__(self):
            self.memories = []

        def add(self, user_id, content):
            memory = {
                "user_id": user_id,
                "content": content,
                "access_count": 0,
                "last_accessed_at": None,
            }
            self.memories.append(memory)
            return memory

        def retrieve(self, user_id, query):
            results = []
            for memory in self.memories:
                if memory["user_id"] == user_id and query.lower() in memory["content"].lower():
                    results.append(memory)
            return results

    return FakeMemoryStore()


def test_user_memory_isolation():
    memory_store = fake_memory_store()
    memory_a = memory_store.add(user_id="user_a", content="User A prefers Python")

    memory_b = memory_store.add(user_id="user_b", content="User B prefers Python")

    results_a = memory_store.retrieve(user_id="user_a", query="Python")

    results_b = memory_store.retrieve(user_id="user_b", query="Python")

    assert memory_a in results_a
    assert memory_b not in results_a

    assert memory_b in results_b
    assert memory_a not in results_b
