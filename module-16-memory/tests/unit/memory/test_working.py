from __future__ import annotations

import asyncio
from datetime import timedelta

from memory_system import ConversationMessage, WorkingMemoryRegistry
from memory_system.memory.working import WorkingMemory
from tests.fakes import CharacterCounter


def test_aggressive_truncation_preserves_pinned_goals() -> None:
    working = WorkingMemory("user-1", "conversation-1", token_counter=CharacterCounter())
    pinned = working.pin_goal("Finish the migration")
    working.add_message(ConversationMessage(role="user", content="Keep this recent"))

    context = working.snapshot(token_budget=0)

    assert pinned in context.pinned_goals
    assert context.messages == []
    assert context.token_count > context.token_budget


async def test_registry_expires_idle_conversations() -> None:
    registry = WorkingMemoryRegistry(max_age=timedelta(milliseconds=10))
    active = registry.get("user-1", "conversation-1")
    await asyncio.sleep(0.02)

    assert registry.expire() == 1
    assert registry.get("user-1", "conversation-1") is not active
