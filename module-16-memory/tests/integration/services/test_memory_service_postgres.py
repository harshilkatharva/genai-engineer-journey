from __future__ import annotations

import os
from importlib.resources import files
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import asyncpg
import pytest

from memory_system import EpisodicMemory, LongTermMemory, MemoryCandidate, WorkingMemoryRegistry
from memory_system.core import Settings
from memory_system.db import PostgresMemoryStore
from memory_system.services.memory_service import MemoryService

SETTINGS = Settings()
TEST_DATABASE_URL = os.getenv("MEMORY_TEST_DATABASE_URL") or SETTINGS.database_url
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not SETTINGS.integration_test,
        reason="Set INTEGRATION_TEST=1 in .env to enable integration tests",
    ),
]


class FixedEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [1.0] + [0.0] * 383


@pytest.mark.asyncio
async def test_memory_service_deletes_persistent_and_working_user_memory() -> None:
    parsed_url = urlsplit(TEST_DATABASE_URL)
    if parsed_url.scheme == "postgresql+psycopg":
        parsed_url = parsed_url._replace(scheme="postgresql")
    database_url = urlunsplit(parsed_url)
    schema = files("memory_system.db").joinpath("schema.sql").read_text()
    connection = await asyncpg.connect(database_url)
    try:
        await connection.execute(schema)
    finally:
        await connection.close()

    store = await PostgresMemoryStore.connect(database_url)
    user_id = f"integration-{uuid4()}"
    settings = Settings(memory_embedding_dimension=384)
    long_term = LongTermMemory(store, FixedEmbedder(), settings=settings)
    episodic = EpisodicMemory(store, settings)
    working = WorkingMemoryRegistry()
    active_context = working.get(user_id, "conversation-1")
    active_context.pin_goal("Keep this conversation private")
    service = MemoryService(long_term, episodic, working)

    try:
        await long_term.store(user_id, MemoryCandidate(content="Prefers green tea"))
        await episodic.record(user_id, "task-1", "send", "report@example.test")

        deleted = await service.delete_all_memories(user_id)

        assert deleted == {"long_term": 1, "episodic": 1}
        assert await service.view_memories(user_id) == []
        assert (
            await episodic.check(user_id, "task-1", "send", "report@example.test")
        ).already_attempted is False
        assert working.get(user_id, "conversation-1") is not active_context
    finally:
        await store.close()
