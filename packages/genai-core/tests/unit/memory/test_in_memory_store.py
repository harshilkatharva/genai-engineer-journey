from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from genai_core.memory.in_memory_store import InMemoryMemoryStore, _cosine_similarity
from genai_core.memory.token_counter import HeuristicTokenCounter
from genai_core.models.memory_model import (
    ActionEpisode,
    ConversationMessage,
    LongTermMemoryRecord,
    MemoryCategory,
    MessageRole,
    PinnedGoal,
)


@pytest.mark.asyncio
async def test_cosine_similarity() -> None:
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert _cosine_similarity([], []) == 0.0
    assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


@pytest.mark.asyncio
async def test_in_memory_short_term_lifecycle() -> None:
    store = InMemoryMemoryStore()
    user_id = "user_1"
    session_id = "sess_1"

    # Append messages
    m1 = ConversationMessage(role=MessageRole.USER, content="Hello")
    m2 = ConversationMessage(role=MessageRole.ASSISTANT, content="Hi there!")
    await store.append_message(user_id, session_id, m1)
    await store.append_message(user_id, session_id, m2)

    messages = await store.get_messages(user_id, session_id)
    assert len(messages) == 2
    assert messages[0].content == "Hello"
    assert messages[1].content == "Hi there!"

    # Test limit
    limited = await store.get_messages(user_id, session_id, limit=1)
    assert len(limited) == 1
    assert limited[0].content == "Hi there!"

    # Pinned goals
    g1 = PinnedGoal(goal="Maintain safety rules")
    await store.set_pinned_goals(user_id, session_id, [g1])
    goals = await store.get_pinned_goals(user_id, session_id)
    assert len(goals) == 1
    assert goals[0].goal == "Maintain safety rules"

    # Pruning snapshot
    counter = HeuristicTokenCounter()
    ctx = await store.prune_session(user_id, session_id, max_tokens=1000, token_counter=counter)
    assert len(ctx.messages) == 2
    assert len(ctx.pinned_goals) == 1
    assert ctx.total_tokens > 0
    assert not ctx.truncated

    # Pruning with very tight budget truncates older messages
    ctx_tight = await store.prune_session(user_id, session_id, max_tokens=15, token_counter=counter)
    assert ctx_tight.truncated
    assert len(ctx_tight.pinned_goals) == 1

    # Clear messages
    cleared = await store.clear_messages(user_id, session_id)
    assert cleared == 2
    assert len(await store.get_messages(user_id, session_id)) == 0

    await store.close()


@pytest.mark.asyncio
async def test_in_memory_user_isolation() -> None:
    store = InMemoryMemoryStore()
    u1, u2 = "user_alpha", "user_beta"
    sess = "sess_shared_name"

    await store.append_message(u1, sess, ConversationMessage(role="user", content="Secret Alpha"))
    await store.append_message(u2, sess, ConversationMessage(role="user", content="Secret Beta"))

    u1_msgs = await store.get_messages(u1, sess)
    u2_msgs = await store.get_messages(u2, sess)

    assert len(u1_msgs) == 1 and u1_msgs[0].content == "Secret Alpha"
    assert len(u2_msgs) == 1 and u2_msgs[0].content == "Secret Beta"

    # Long-term isolation
    rec1 = LongTermMemoryRecord(user_id=u1, content="Alpha fact", embedding=[1.0, 0.0])
    rec2 = LongTermMemoryRecord(user_id=u2, content="Beta fact", embedding=[0.0, 1.0])
    await store.save_record(rec1)
    await store.save_record(rec2)

    assert await store.get_record(u1, rec2.id) is None
    assert await store.get_record(u2, rec1.id) is None

    u1_search = await store.search_records(u1, query_embedding=[0.0, 1.0])
    assert len(u1_search) == 0 or all(r.record.user_id == u1 for r in u1_search)

    await store.close()


@pytest.mark.asyncio
async def test_in_memory_long_term_search_recency_and_supersession() -> None:
    store = InMemoryMemoryStore()
    user_id = "user_recency"
    now = datetime.now(UTC)

    # Record 1: Old fact, superseded
    rec_old = LongTermMemoryRecord(
        user_id=user_id,
        content="I live in Boston",
        category=MemoryCategory.FACT,
        created_at=now - timedelta(days=60),
        embedding=[1.0, 0.0],
    )
    await store.save_record(rec_old)

    # Record 2: New fact that supersedes Record 1
    rec_new = LongTermMemoryRecord(
        user_id=user_id,
        content="I moved to Seattle",
        category=MemoryCategory.FACT,
        supersedes_ids=[rec_old.id],
        created_at=now - timedelta(days=1),
        embedding=[0.9, 0.1],
    )
    await store.save_record(rec_new)

    # Search with query close to both
    results = await store.search_records(
        user_id=user_id,
        query_embedding=[1.0, 0.0],
        top_k=5,
        half_life_days=30,
        include_superseded=False,
    )

    # rec_old should be excluded because it's superseded
    assert len(results) == 1
    assert results[0].record.id == rec_new.id
    assert not results[0].record.is_superseded
    assert results[0].recency_score > 0.9

    # Search with include_superseded=True should return both, with rec_new ranked higher
    results_all = await store.search_records(
        user_id=user_id,
        query_embedding=[1.0, 0.0],
        top_k=5,
        half_life_days=30,
        include_superseded=True,
    )
    assert len(results_all) == 2
    assert results_all[0].record.id == rec_new.id
    assert results_all[1].record.id == rec_old.id
    assert results_all[1].record.is_superseded

    await store.close()


@pytest.mark.asyncio
async def test_in_memory_episodic_history_and_purge() -> None:
    store = InMemoryMemoryStore()
    user_id = "user_ep"
    now = datetime.now(UTC)

    ep1 = ActionEpisode(
        user_id=user_id,
        action_type="send_email",
        parameters={"to": "test@example.com"},
        parameter_hash="hash_123",
        occurred_at=now - timedelta(days=10),
    )
    ep2 = ActionEpisode(
        user_id=user_id,
        action_type="send_email",
        parameters={"to": "test@example.com"},
        parameter_hash="hash_123",
        occurred_at=now - timedelta(days=1),
    )
    await store.record_episode(ep1)
    await store.record_episode(ep2)

    found = await store.find_matching_episode(user_id, "send_email", "hash_123")
    assert found is not None
    # Most recent returned
    assert found.id == ep2.id

    episodes = await store.get_episodes(user_id, action_type="send_email")
    assert len(episodes) == 2

    # Purge older than 5 days
    purged = await store.purge_episodes(older_than=now - timedelta(days=5), user_id=user_id)
    assert purged == 1

    remaining = await store.get_episodes(user_id)
    assert len(remaining) == 1
    assert remaining[0].id == ep2.id

    await store.close()
