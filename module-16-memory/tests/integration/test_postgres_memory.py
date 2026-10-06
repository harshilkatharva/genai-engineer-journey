from __future__ import annotations

import os
from importlib.resources import files
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import asyncpg
import pytest

from memory_system import (
    EpisodicMemory,
    LongTermMemory,
    MemoryCandidate,
    MemoryService,
)
from memory_system.core import Settings
from memory_system.db import PostgresMemoryStore

TEST_DATABASE_URL = os.getenv("MEMORY_TEST_DATABASE_URL") or Settings().database_url
pytestmark = [
    pytest.mark.integration,
]


class FixedEmbedder:
    # Supplies a stable pgvector-compatible embedding during the database test.
    async def embed(self, text: str) -> list[float]:
        return [1.0] + [0.0] * 383


# Verifies PostgreSQL memory ranking and user deletion end to end.
@pytest.mark.asyncio
async def test_postgres_supersession_and_user_deletion() -> None:
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
    service = MemoryService(long_term, episodic)
    try:
        old = await long_term.store(
            user_id, MemoryCandidate(content="User lives in Paris", category="fact")
        )
        new = await long_term.store(
            user_id,
            MemoryCandidate(
                content="User moved to Berlin",
                category="fact",
                supersedes_ids=[old.id],
            ),
        )
        await episodic.record(user_id, "task-1", "send", "report@example.test", status="completed")

        results = await long_term.retrieve(user_id, "Where does the user live?", limit=2)
        assert results[0].memory.id == new.id
        assert results[1].superseded is True
        assert await service.delete_memory(user_id, old.id) is True
        assert all(
            result.memory.id != old.id
            for result in await long_term.retrieve(user_id, "Paris", limit=2)
        )

        deleted = await service.delete_all_memories(user_id)
        assert deleted == {"long_term": 1, "episodic": 1}
        assert await service.view_memories(user_id) == []
        assert await long_term.retrieve(user_id, "Berlin") == []
    finally:
        await store.close()
