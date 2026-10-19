from __future__ import annotations

import pytest

from genai_core.exceptions import MemoryValidationError
from genai_core.memory.in_memory_store import InMemoryMemoryStore
from genai_core.memory.short_term import ShortTermMemory
from genai_core.models.memory_model import MessageRole


@pytest.mark.asyncio
async def test_short_term_memory_messages_and_tokens() -> None:
    store = InMemoryMemoryStore()
    stm = ShortTermMemory(store=store)

    user_id = "test_user"
    session_id = "test_session"

    msg = await stm.add_message(
        user_id=user_id,
        session_id=session_id,
        role=MessageRole.USER,
        content="What is the weather today?",
    )
    assert msg.token_count is not None
    assert msg.token_count > 0

    history = await stm.get_messages(user_id=user_id, session_id=session_id)
    assert len(history) == 1
    assert history[0].content == "What is the weather today?"

    # Pinning goals
    pinned = await stm.pin_goal(
        user_id=user_id,
        session_id=session_id,
        goal="Respond in French only",
    )
    assert pinned.goal == "Respond in French only"

    goals = await stm.get_pinned_goals(user_id=user_id, session_id=session_id)
    assert len(goals) == 1

    # Snapshot budgeting
    ctx = await stm.snapshot(user_id=user_id, session_id=session_id, token_budget=500)
    assert len(ctx.messages) == 1
    assert len(ctx.pinned_goals) == 1
    assert not ctx.truncated

    # Unpin goal
    unpinned = await stm.unpin_goal(user_id=user_id, session_id=session_id, goal_id=pinned.id)
    assert unpinned
    assert len(await stm.get_pinned_goals(user_id=user_id, session_id=session_id)) == 0

    # Validation errors
    with pytest.raises(MemoryValidationError):
        await stm.add_message(user_id="", session_id="s", role="user", content="hi")

    with pytest.raises(MemoryValidationError):
        await stm.add_message(user_id="u", session_id="s", role="user", content="")

    with pytest.raises(MemoryValidationError):
        await stm.pin_goal(user_id="u", session_id="s", goal="   ")

    with pytest.raises(MemoryValidationError):
        await stm.snapshot(user_id="u", session_id="s", token_budget=-1)

    await store.close()
