from __future__ import annotations

from typing import Any

from ..core import get_settings
from ..exceptions import MemoryValidationError
from ..models.memory_model import (
    ConversationMessage,
    MessageRole,
    PinnedGoal,
    WorkingContext,
)
from ..observability import get_logger
from .protocol import MemoryStorageAdapter, TokenCounter
from .token_counter import get_default_token_counter

_logger = get_logger("genai_core.memory.short_term")


class ShortTermMemory:
    """Manages conversational session history, pinned agent goals, and token budget snapshots."""

    def __init__(
        self,
        store: MemoryStorageAdapter,
        token_counter: TokenCounter | None = None,
        ttl_hours: int | None = None,
    ) -> None:
        self.store = store
        self.token_counter = token_counter or get_default_token_counter()
        settings = get_settings()
        self.ttl_hours = ttl_hours if ttl_hours is not None else settings.memory_working_ttl_hours

    async def add_message(
        self,
        user_id: str,
        session_id: str,
        role: MessageRole | str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationMessage:
        """Add a turn message to the working context with computed token count."""
        if not user_id or not session_id:
            raise MemoryValidationError("user_id and session_id are required")
        if not content:
            raise MemoryValidationError("Message content cannot be empty")

        msg = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        msg.token_count = self.token_counter.count_message(msg)
        await self.store.append_message(user_id, session_id, msg)
        return msg

    async def get_messages(
        self, user_id: str, session_id: str, limit: int | None = None
    ) -> list[ConversationMessage]:
        """Retrieve recent conversation messages in chronological order."""
        if not user_id or not session_id:
            raise MemoryValidationError("user_id and session_id are required")
        return await self.store.get_messages(user_id, session_id, limit)

    async def pin_goal(
        self,
        user_id: str,
        session_id: str,
        goal: str,
        metadata: dict[str, Any] | None = None,
    ) -> PinnedGoal:
        """Pin a system instruction or goal that persists through pruning."""
        if not user_id or not session_id:
            raise MemoryValidationError("user_id and session_id are required")
        if not goal.strip():
            raise MemoryValidationError("Goal cannot be empty")

        pinned = PinnedGoal(goal=goal.strip(), metadata=metadata or {})
        existing = await self.store.get_pinned_goals(user_id, session_id)
        existing.append(pinned)
        await self.store.set_pinned_goals(user_id, session_id, existing)
        return pinned

    async def unpin_goal(self, user_id: str, session_id: str, goal_id: str) -> bool:
        """Unpin a previously registered goal by ID."""
        if not user_id or not session_id:
            raise MemoryValidationError("user_id and session_id are required")
        existing = await self.store.get_pinned_goals(user_id, session_id)
        remaining = [g for g in existing if g.id != goal_id]
        if len(remaining) != len(existing):
            await self.store.set_pinned_goals(user_id, session_id, remaining)
            return True
        return False

    async def get_pinned_goals(self, user_id: str, session_id: str) -> list[PinnedGoal]:
        """Retrieve all active pinned goals for the session."""
        if not user_id or not session_id:
            raise MemoryValidationError("user_id and session_id are required")
        return await self.store.get_pinned_goals(user_id, session_id)

    async def snapshot(
        self,
        user_id: str,
        session_id: str,
        token_budget: int | None = None,
    ) -> WorkingContext:
        """Assemble a budgeted working context preserving pinned goals and latest messages."""
        if not user_id or not session_id:
            raise MemoryValidationError("user_id and session_id are required")
        budget = token_budget if token_budget is not None else get_settings().memory_token_budget
        if budget <= 0:
            raise MemoryValidationError("token_budget must be a positive integer")
        return await self.store.prune_session(user_id, session_id, budget, self.token_counter)

    async def clear(self, user_id: str, session_id: str) -> None:
        """Clear all messages and pinned goals for the session."""
        if not user_id or not session_id:
            raise MemoryValidationError("user_id and session_id are required")
        await self.store.clear_session(user_id, session_id)

    async def is_expired(self, user_id: str, session_id: str) -> bool:
        """Check whether the session has exceeded inactivity TTL."""
        if not user_id or not session_id:
            raise MemoryValidationError("user_id and session_id are required")
        max_age_seconds = self.ttl_hours * 3600.0
        return await self.store.is_session_expired(user_id, session_id, max_age_seconds)
