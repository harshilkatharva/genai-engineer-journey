from __future__ import annotations

import contextlib
import os
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta

import pytest

from genai_core.memory.postgres_store import PostgresMemoryStore
from genai_core.memory.token_counter import HeuristicTokenCounter
from genai_core.models.memory_model import (
    ActionEpisode,
    ConversationMessage,
    LongTermMemoryRecord,
    MemoryCategory,
    MessageRole,
    PinnedGoal,
)

POSTGRES_URL = (
    os.getenv("DATABASE_CONNECTION_CONVERSATION_URL")
    or os.getenv("POSTGRES_URL")
    or "postgresql://postgres:postgres@127.0.0.1:5432/ai_search"
)

TEST_PREFIX = "test_mem_int_"


@pytest.fixture
async def pg_store() -> AsyncGenerator[PostgresMemoryStore, None]:
    try:
        import psycopg  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("psycopg is not installed")

    try:
        conn = await psycopg.AsyncConnection.connect(
            POSTGRES_URL, autocommit=True, connect_timeout=3
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Live PostgreSQL instance not reachable at {POSTGRES_URL}: {exc}")

    store = PostgresMemoryStore(
        dimensions=4,
        connection=conn,
        table_prefix=TEST_PREFIX,
    )

    try:
        await store.initialize()
        yield store
    finally:
        with contextlib.suppress(Exception):
            async with conn.cursor() as cur:
                await cur.execute(f"DROP TABLE IF EXISTS {store.short_term_messages_table};")
                await cur.execute(f"DROP TABLE IF EXISTS {store.short_term_goals_table};")
                await cur.execute(f"DROP TABLE IF EXISTS {store.short_term_sessions_table};")
                await cur.execute(f"DROP TABLE IF EXISTS {store.long_term_table};")
                await cur.execute(f"DROP TABLE IF EXISTS {store.episodic_table};")
            await conn.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_postgres_short_term_lifecycle(pg_store: PostgresMemoryStore) -> None:
    user_id = "pg_user_1"
    session_id = "pg_sess_1"

    # 1. Append messages
    m1 = ConversationMessage(role=MessageRole.USER, content="Hello from PostgreSQL test!")
    m2 = ConversationMessage(role=MessageRole.ASSISTANT, content="PostgreSQL memory is active.")
    await pg_store.append_message(user_id, session_id, m1)
    await pg_store.append_message(user_id, session_id, m2)

    messages = await pg_store.get_messages(user_id, session_id)
    assert len(messages) == 2
    assert messages[0].content == "Hello from PostgreSQL test!"
    assert messages[1].content == "PostgreSQL memory is active."

    # 2. Pinned goals
    g1 = PinnedGoal(goal="Maintain idempotency across agent tasks")
    await pg_store.set_pinned_goals(user_id, session_id, [g1])
    goals = await pg_store.get_pinned_goals(user_id, session_id)
    assert len(goals) == 1
    assert goals[0].goal == "Maintain idempotency across agent tasks"

    # 3. Pruning snapshot
    counter = HeuristicTokenCounter()
    ctx = await pg_store.prune_session(user_id, session_id, max_tokens=1000, token_counter=counter)
    assert len(ctx.messages) == 2
    assert len(ctx.pinned_goals) == 1
    assert not ctx.truncated

    # 4. Clear session
    await pg_store.clear_session(user_id, session_id)
    assert len(await pg_store.get_messages(user_id, session_id)) == 0
    assert len(await pg_store.get_pinned_goals(user_id, session_id)) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_postgres_long_term_persistence_and_search(pg_store: PostgresMemoryStore) -> None:
    user_id = "pg_user_lt"
    now = datetime.now(UTC)

    # Record 1: Old fact, 4D vector
    rec1 = LongTermMemoryRecord(
        user_id=user_id,
        content="I prefer dark mode UI interfaces",
        category=MemoryCategory.PREFERENCE,
        created_at=now - timedelta(days=20),
        embedding=[1.0, 0.0, 0.0, 0.0],
    )
    await pg_store.save_record(rec1)

    # Record 2: New fact that supersedes Record 1
    rec2 = LongTermMemoryRecord(
        user_id=user_id,
        content="I switched to high-contrast light mode",
        category=MemoryCategory.PREFERENCE,
        supersedes_ids=[rec1.id],
        created_at=now - timedelta(days=1),
        embedding=[0.9, 0.1, 0.0, 0.0],
    )
    await pg_store.save_record(rec2)

    # Search with query vector aligned with both
    results = await pg_store.search_records(
        user_id=user_id,
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        top_k=5,
        half_life_days=30,
        include_superseded=False,
    )

    # Superseded record excluded
    assert len(results) == 1
    assert results[0].record.id == rec2.id
    assert not results[0].record.is_superseded
    assert results[0].similarity > 0.8

    # Include superseded
    all_results = await pg_store.search_records(
        user_id=user_id,
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        top_k=5,
        half_life_days=30,
        include_superseded=True,
    )
    assert len(all_results) == 2
    assert all_results[0].record.id == rec2.id
    assert all_results[1].record.id == rec1.id
    assert all_results[1].record.is_superseded

    # Delete
    deleted = await pg_store.delete_record(user_id=user_id, record_id=rec1.id)
    assert deleted
    assert await pg_store.get_record(user_id, rec1.id) is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_postgres_episodic_action_history(pg_store: PostgresMemoryStore) -> None:
    user_id = "pg_user_ep"
    now = datetime.now(UTC)

    ep1 = ActionEpisode(
        user_id=user_id,
        action_type="deploy_service",
        parameters={"env": "staging", "version": "v1.0.0"},
        parameter_hash="sha_staging_v1",
        occurred_at=now - timedelta(days=10),
    )
    ep2 = ActionEpisode(
        user_id=user_id,
        action_type="deploy_service",
        parameters={"env": "staging", "version": "v1.0.1"},
        parameter_hash="sha_staging_v2",
        occurred_at=now - timedelta(days=1),
    )
    await pg_store.record_episode(ep1)
    await pg_store.record_episode(ep2)

    found = await pg_store.find_matching_episode(
        user_id=user_id,
        action_type="deploy_service",
        parameter_hash="sha_staging_v2",
    )
    assert found is not None
    assert found.parameters["version"] == "v1.0.1"

    # Purge older than 5 days
    purged = await pg_store.purge_episodes(older_than=now - timedelta(days=5), user_id=user_id)
    assert purged == 1

    remaining = await pg_store.get_episodes(user_id=user_id)
    assert len(remaining) == 1
    assert remaining[0].parameter_hash == "sha_staging_v2"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_postgres_strict_user_isolation(pg_store: PostgresMemoryStore) -> None:
    u1, u2 = "pg_isolation_user_a", "pg_isolation_user_b"
    sess = "sess_iso"

    # Short term isolation
    await pg_store.append_message(u1, sess, ConversationMessage(role="user", content="Secret A"))
    await pg_store.append_message(u2, sess, ConversationMessage(role="user", content="Secret B"))

    u1_msgs = await pg_store.get_messages(u1, sess)
    u2_msgs = await pg_store.get_messages(u2, sess)
    assert len(u1_msgs) == 1 and u1_msgs[0].content == "Secret A"
    assert len(u2_msgs) == 1 and u2_msgs[0].content == "Secret B"

    # Long term isolation
    rec_a = LongTermMemoryRecord(user_id=u1, content="A profile", embedding=[1.0, 0.0, 0.0, 0.0])
    rec_b = LongTermMemoryRecord(user_id=u2, content="B profile", embedding=[0.0, 1.0, 0.0, 0.0])
    await pg_store.save_record(rec_a)
    await pg_store.save_record(rec_b)

    assert await pg_store.get_record(u1, rec_b.id) is None
    assert await pg_store.get_record(u2, rec_a.id) is None

    u1_search = await pg_store.search_records(u1, query_embedding=[0.0, 1.0, 0.0, 0.0])
    assert all(r.record.user_id == u1 for r in u1_search)

    # Episodic isolation
    ep_a = ActionEpisode(user_id=u1, action_type="test", parameter_hash="hash_a")
    ep_b = ActionEpisode(user_id=u2, action_type="test", parameter_hash="hash_a")
    await pg_store.record_episode(ep_a)
    await pg_store.record_episode(ep_b)

    assert await pg_store.find_matching_episode(u1, "test", "hash_a") is not None
    episodes_u1 = await pg_store.get_episodes(u1)
    assert all(e.user_id == u1 for e in episodes_u1)
