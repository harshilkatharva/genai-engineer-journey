from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from memory_system import ConversationMessage, WorkingMemoryRegistry
from memory_system.memory import working as working_module
from memory_system.memory.working import WorkingMemory
from tests.fakes import CharacterCounter


def test_working_memory_snapshot_keeps_recent_fit_messages_and_validates_budget() -> None:
    for user_id, conversation_id, max_age in (
        ("", "conversation", timedelta(hours=1)),
        ("user", "", timedelta(hours=1)),
        ("user", "conversation", timedelta(0)),
    ):
        with pytest.raises(ValueError):
            WorkingMemory(
                user_id,
                conversation_id,
                token_counter=CharacterCounter(),
                max_age=max_age,
            )

    memory = WorkingMemory("user", "conversation", token_counter=CharacterCounter())
    memory.add_message(ConversationMessage(role="user", content="old"))
    memory.add_message(ConversationMessage(role="assistant", content="new"))

    context = memory.snapshot(token_budget=20)
    assert [message.content for message in context.messages] == ["new"]
    assert context.token_count <= context.token_budget
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError, match="non-negative integer"):
            memory.snapshot(token_budget=invalid)  # type: ignore[arg-type]


def test_working_memory_pin_unpin_clear_and_expiration_use_last_access() -> None:
    current = [datetime(2030, 1, 1, tzinfo=UTC)]
    original_now = working_module.utc_now
    working_module.utc_now = lambda: current[0]
    try:
        memory = WorkingMemory(
            "user", "conversation", token_counter=CharacterCounter(), max_age=timedelta(hours=1)
        )
        goal = memory.pin_goal("Do not forget")
        pinned_context = memory.snapshot(token_budget=0)
        assert pinned_context.pinned_goals == [goal]
        assert pinned_context.messages == []
        assert pinned_context.token_count > pinned_context.token_budget
        assert memory.unpin_goal(goal.id) is True
        assert memory.unpin_goal(goal.id) is False
        assert memory.is_expired(current[0] + timedelta(hours=1)) is True

        memory.add_message(ConversationMessage(role="user", content="hello"))
        assert memory.is_expired(current[0] + timedelta(minutes=30)) is False
        memory.clear()
        assert memory.snapshot(token_budget=10).messages == []
    finally:
        working_module.utc_now = original_now


def test_working_memory_registry_reuses_expires_and_clears_scoped_contexts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = [datetime(2030, 1, 1, tzinfo=UTC)]
    monkeypatch.setattr(working_module, "utc_now", lambda: current[0])
    registry = WorkingMemoryRegistry(max_age=timedelta(hours=1))
    first = registry.get("user-1", "conversation-1")

    assert registry.get("user-1", "conversation-1") is first
    registry.get("user-1", "conversation-2")
    registry.get("user-2", "conversation-1")
    current[0] += timedelta(hours=1)
    assert registry.expire() == 3
    assert registry.get("user-1", "conversation-1") is not first
    registry.clear_conversation("user-1", "conversation-1")
    registry.clear_user("user-1")
    assert registry.expire() == 0
