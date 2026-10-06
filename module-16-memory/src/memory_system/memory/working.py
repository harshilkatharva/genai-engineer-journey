from __future__ import annotations

from datetime import datetime, timedelta
from typing import Protocol
from uuid import UUID

import tiktoken

from ..core import get_settings
from ..models import ConversationMessage, PinnedGoal, WorkingContext, utc_now


class TokenCounter(Protocol):
    # Returns the number of model tokens in the supplied text.
    def count(self, text: str) -> int: ...


class TiktokenCounter:
    # Selects the tokenizer encoding used for working-memory budgeting.
    def __init__(self, encoding: str = "cl100k_base") -> None:
        self.encoding = tiktoken.get_encoding(encoding)

    # Counts text tokens using the configured tokenizer.
    def count(self, text: str) -> int:
        return len(self.encoding.encode(text))


class WorkingMemory:
    # Initializes conversation-scoped memory with its age and token policies.
    def __init__(
        self,
        user_id: str,
        conversation_id: str,
        token_counter: TokenCounter | None = None,
        max_age: timedelta | None = None,
    ) -> None:
        if max_age is None:
            max_age = timedelta(hours=get_settings().memory_working_ttl_hours)
        if not user_id or not conversation_id:
            raise ValueError("user_id and conversation_id are required")
        if max_age <= timedelta(0):
            raise ValueError("max_age must be positive")
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.max_age = max_age
        self._token_counter = token_counter or TiktokenCounter()
        self._messages: list[ConversationMessage] = []
        self._pinned_goals: list[PinnedGoal] = []
        self._created_at = utc_now()
        self._last_accessed_at = self._created_at

    # Returns the time this conversation's working memory was last accessed.
    @property
    def last_accessed_at(self) -> datetime:
        return self._last_accessed_at

    # Adds a conversation message to the working context.
    def add_message(self, message: ConversationMessage) -> None:
        self._messages.append(message)
        self._last_accessed_at = utc_now()

    # Pins an instruction so it remains present after context truncation.
    def pin_goal(self, content: str) -> PinnedGoal:
        goal = PinnedGoal(content=content)
        self._pinned_goals.append(goal)
        self._last_accessed_at = utc_now()
        return goal

    # Removes a pinned goal by its identifier.
    def unpin_goal(self, goal_id: UUID) -> bool:
        remaining = [goal for goal in self._pinned_goals if goal.id != goal_id]
        changed = len(remaining) != len(self._pinned_goals)
        self._pinned_goals = remaining
        self._last_accessed_at = utc_now()
        return changed

    # Builds a token-budgeted context while always retaining pinned goals.
    def snapshot(self, token_budget: int | None = None) -> WorkingContext:
        if token_budget is None:
            token_budget = get_settings().memory_token_budget
        if isinstance(token_budget, bool) or not isinstance(token_budget, int) or token_budget < 0:
            raise ValueError("token_budget must be a non-negative integer")
        self._last_accessed_at = utc_now()
        pinned_cost = sum(
            self._token_counter.count(f"Pinned instruction: {goal.content}\n")
            for goal in self._pinned_goals
        )
        remaining = token_budget - pinned_cost
        retained_reversed: list[ConversationMessage] = []
        message_cost = 0
        if remaining > 0:
            for message in reversed(self._messages):
                cost = self._token_counter.count(f"{message.role}: {message.content}\n")
                if cost > remaining:
                    break
                retained_reversed.append(message)
                remaining -= cost
                message_cost += cost
        retained = list(reversed(retained_reversed))
        total = pinned_cost + message_cost
        return WorkingContext(
            pinned_goals=self._pinned_goals.copy(),
            messages=retained,
            token_count=total,
            token_budget=token_budget,
        )

    # Clears all messages and pinned goals from this conversation.
    def clear(self) -> None:
        self._messages.clear()
        self._pinned_goals.clear()
        self._last_accessed_at = utc_now()

    # Reports whether this conversation has exceeded its inactivity lifetime.
    def is_expired(self, now: datetime | None = None) -> bool:
        current = now or utc_now()
        return current - self._last_accessed_at >= self.max_age


class WorkingMemoryRegistry:
    # Creates a process-local registry for conversation working memories.
    def __init__(self, max_age: timedelta | None = None) -> None:
        if max_age is None:
            max_age = timedelta(hours=get_settings().memory_working_ttl_hours)
        if max_age <= timedelta(0):
            raise ValueError("max_age must be positive")
        self._max_age = max_age
        self._conversations: dict[tuple[str, str], WorkingMemory] = {}

    # Returns the active memory for a conversation, creating it when needed.
    def get(self, user_id: str, conversation_id: str) -> WorkingMemory:
        key = (user_id, conversation_id)
        memory = self._conversations.get(key)
        if memory is None or memory.is_expired():
            memory = WorkingMemory(user_id, conversation_id, max_age=self._max_age)
            self._conversations[key] = memory
        return memory

    # Removes one conversation's working memory from the registry.
    def clear_conversation(self, user_id: str, conversation_id: str) -> None:
        self._conversations.pop((user_id, conversation_id), None)

    # Removes every active conversation belonging to a user.
    def clear_user(self, user_id: str) -> None:
        self._conversations = {
            key: memory for key, memory in self._conversations.items() if key[0] != user_id
        }

    # Removes expired conversations and returns the number cleared.
    def expire(self, now: datetime | None = None) -> int:
        expired = [key for key, memory in self._conversations.items() if memory.is_expired(now)]
        for key in expired:
            del self._conversations[key]
        return len(expired)
