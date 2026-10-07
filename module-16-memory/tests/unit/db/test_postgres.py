from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest

from memory_system.db import postgres
from memory_system.db.postgres import PostgresMemoryStore
from memory_system.models import ActionEpisode, LongTermMemoryRecord


def make_memory(**overrides: Any) -> LongTermMemoryRecord:
    values: dict[str, Any] = {
        "user_id": "user-1",
        "content": "Likes tea",
        "category": "preference",
        "supersedes_ids": [],
        "created_at": datetime.now(UTC),
        "expires_at": None,
    }
    values.update(overrides)
    return LongTermMemoryRecord(**values)


def make_episode(**overrides: Any) -> ActionEpisode:
    now = datetime.now(UTC)
    values: dict[str, Any] = {
        "user_id": "user-1",
        "task_id": "task-1",
        "action_type": "send_email",
        "target": "team",
        "parameters": {"subject": "status"},
        "occurred_at": now,
        "expires_at": now + timedelta(days=1),
    }
    values.update(overrides)
    return ActionEpisode(**values)


class FakePool:
    def __init__(self, connection: Any) -> None:
        self.connection = connection
        self.close = _async_mock()

    @asynccontextmanager
    async def acquire(self):
        yield self.connection


def _async_mock(*args: Any, **kwargs: Any):
    from unittest.mock import AsyncMock

    return AsyncMock(*args, **kwargs)


def make_connection() -> Any:
    from unittest.mock import AsyncMock, MagicMock

    connection = MagicMock()
    connection.execute = AsyncMock(return_value="DELETE 1")
    connection.fetch = AsyncMock(return_value=[])
    connection.fetchrow = AsyncMock(return_value=None)
    connection.fetchval = AsyncMock(return_value=0)
    transaction = MagicMock()
    transaction.__aenter__ = AsyncMock(return_value=None)
    transaction.__aexit__ = AsyncMock(return_value=None)
    connection.transaction.return_value = transaction
    return connection


@pytest.mark.asyncio
async def test_connect_validates_url_and_initializes_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = FakePool(make_connection())
    create_pool = _async_mock(return_value=pool)
    monkeypatch.setattr(postgres.asyncpg, "create_pool", create_pool)
    monkeypatch.setattr(postgres, "register_vector", "register")

    with pytest.raises(ValueError, match="DATABASE_URL"):
        await PostgresMemoryStore.connect("")
    with pytest.raises(RuntimeError, match="initialization failed"):
        create_pool.return_value = None
        await PostgresMemoryStore.connect("postgresql://db")
    create_pool.return_value = pool
    store = await PostgresMemoryStore.connect("postgresql://db", min_size=2, max_size=4)

    assert store.pool is pool
    create_pool.assert_awaited_with("postgresql://db", min_size=2, max_size=4, init="register")


@pytest.mark.asyncio
async def test_close_saves_memory_and_lists_records() -> None:
    connection = make_connection()
    pool = FakePool(connection)
    store = PostgresMemoryStore(pool)
    memory = make_memory()
    row = memory.model_dump()
    connection.fetch.return_value = [row]

    await store.close()
    saved = await store.save_memory(memory, [0.1, 0.2])
    listed = await store.list_memories("user-1", 5)

    assert saved is memory
    assert listed == [memory]
    pool.close.assert_awaited_once()
    connection.execute.assert_awaited_once()
    assert connection.fetch.await_args.args[1:] == ("user-1", 5)


@pytest.mark.asyncio
async def test_search_maps_results_and_bounds_candidates() -> None:
    connection = make_connection()
    record = make_memory()
    connection.fetch.return_value = [
        {
            **record.model_dump(),
            "similarity": 0.8,
            "rank_score": 0.7,
            "is_superseded": True,
        }
    ]
    store = PostgresMemoryStore(FakePool(connection))

    for limit, candidate_limit in ((1, 100), (20, 200), (500, 5000)):
        results = await store.search_memories("user-1", [1.0, 0.0], limit, 30)
        assert len(results) == 1
        assert results[0].memory == record
        assert results[0].similarity == 0.8
        assert results[0].rank_score == 0.7
        assert results[0].superseded is True
        assert connection.fetch.await_args.args[3:] == (candidate_limit, 30, limit)


@pytest.mark.asyncio
async def test_delete_methods_parse_affected_row_counts() -> None:
    connection = make_connection()
    store = PostgresMemoryStore(FakePool(connection))

    connection.execute.return_value = "DELETE 1"
    assert await store.delete_memory("user-1", uuid4()) is True
    connection.execute.return_value = "DELETE 0"
    assert await store.delete_memory("user-1", uuid4()) is False
    connection.execute.return_value = "DELETE 12"
    assert await store.delete_episodes("user-1") == 12
    connection.execute.return_value = "DELETE 0"
    assert await store.delete_episodes("user-1") == 0


@pytest.mark.asyncio
async def test_delete_all_and_purge_return_layer_counts() -> None:
    connection = make_connection()
    connection.fetchval.side_effect = [3, 2, 4, 1]
    store = PostgresMemoryStore(FakePool(connection))

    assert await store.delete_all_user_data("user-1") == (3, 2)
    assert await store.purge_expired() == (4, 1)
    assert connection.transaction.call_count == 2


@pytest.mark.asyncio
async def test_episode_persistence_and_lookup_validate_json() -> None:
    connection = make_connection()
    store = PostgresMemoryStore(FakePool(connection))
    episode = make_episode()

    assert await store.record_episode(episode, "fingerprint") is episode
    assert json.loads(connection.execute.await_args.args[7]) == episode.parameters
    with pytest.raises(ValueError, match="JSON-serializable"):
        await store.record_episode(make_episode(parameters={"value": float("nan")}), "bad")
    row = {**episode.model_dump(), "parameters": json.dumps(episode.parameters)}
    connection.fetchrow.return_value = row

    found = await store.find_episode("user-1", "task-1", "fingerprint")
    connection.fetchrow.return_value = None

    assert found == episode
    assert found is not None and found.parameters == episode.parameters
    assert await store.find_episode("user-1", "task-1", "missing") is None
