from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from ..models.memory_model import (
    ActionEpisode,
    ConversationMessage,
    LongTermMemoryRecord,
    MemoryCategory,
    MemorySearchResult,
    PinnedGoal,
    WorkingContext,
)


class TokenCounter(Protocol):
    """Protocol for calculating token counts in working memory."""

    def count(self, text: str) -> int:
        """Count tokens in arbitrary text."""
        ...

    def count_message(self, message: ConversationMessage) -> int:
        """Count tokens in a ConversationMessage including role overhead."""
        ...


@runtime_checkable
class MemoryStorageAdapter(Protocol):
    """Protocol for pluggable memory persistence backends (in-memory, PostgreSQL, etc.)."""

    # --- Short-Term / Working Memory ---

    async def append_message(
        self, user_id: str, session_id: str, message: ConversationMessage
    ) -> None:
        """Append a message to the working context of a session."""
        ...

    async def get_messages(
        self, user_id: str, session_id: str, limit: int | None = None
    ) -> list[ConversationMessage]:
        """Retrieve chronological messages for a user session."""
        ...

    async def clear_messages(self, user_id: str, session_id: str) -> int:
        """Remove all messages for a session and return count removed."""
        ...

    async def set_pinned_goals(
        self, user_id: str, session_id: str, goals: list[PinnedGoal]
    ) -> None:
        """Set or update the pinned goals for a session."""
        ...

    async def get_pinned_goals(self, user_id: str, session_id: str) -> list[PinnedGoal]:
        """Retrieve the pinned goals for a session."""
        ...

    async def prune_session(
        self, user_id: str, session_id: str, max_tokens: int, token_counter: TokenCounter
    ) -> WorkingContext:
        """Prune messages to fit within token budget while preserving pinned goals."""
        ...

    async def clear_session(self, user_id: str, session_id: str) -> None:
        """Clear all messages and pinned goals for a specific session."""
        ...

    async def is_session_expired(
        self, user_id: str, session_id: str, max_age_seconds: float
    ) -> bool:
        """Check whether the session has exceeded inactivity TTL."""
        ...

    # --- Long-Term Memory ---

    async def save_record(self, record: LongTermMemoryRecord) -> LongTermMemoryRecord:
        """Save a long-term memory record with its embedding."""
        ...

    async def get_record(self, user_id: str, record_id: str) -> LongTermMemoryRecord | None:
        """Retrieve a specific long-term memory record by ID."""
        ...

    async def list_records(
        self,
        user_id: str,
        category: MemoryCategory | str | None = None,
        include_superseded: bool = False,
        include_expired: bool = False,
        limit: int = 100,
    ) -> list[LongTermMemoryRecord]:
        """List long-term memory records for a user with optional filters."""
        ...

    async def search_records(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 5,
        half_life_days: int = 30,
        similarity_threshold: float = 0.0,
        category: MemoryCategory | str | None = None,
        include_superseded: bool = False,
    ) -> list[MemorySearchResult]:
        """Search user memories using cosine similarity and exponential recency decay."""
        ...

    async def supersede_records(self, user_id: str, superseded_ids: list[str]) -> int:
        """Mark memory records as superseded for a given user."""
        ...

    async def delete_record(self, user_id: str, record_id: str) -> bool:
        """Delete a specific long-term memory record."""
        ...

    async def purge_expired(self, user_id: str | None = None) -> int:
        """Purge expired long-term memory records (across all users or for a specific user)."""
        ...

    # --- Episodic Memory ---

    async def record_episode(self, episode: ActionEpisode) -> ActionEpisode:
        """Record an episodic action event."""
        ...

    async def get_episodes(
        self,
        user_id: str,
        action_type: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[ActionEpisode]:
        """Retrieve episodic action records matching filters."""
        ...

    async def find_matching_episode(
        self,
        user_id: str,
        action_type: str,
        parameter_hash: str,
        session_id: str | None = None,
    ) -> ActionEpisode | None:
        """Find an unexpired episode matching action_type and canonical parameter hash."""
        ...

    async def delete_episodes(self, user_id: str) -> int:
        """Delete all episodic memory records for a user."""
        ...

    async def purge_episodes(self, older_than: datetime, user_id: str | None = None) -> int:
        """Purge episodes older than a given timestamp."""
        ...

    # --- Lifecycle ---

    async def initialize(self) -> None:
        """Initialize backend resources, schemas, or connections if needed."""
        ...

    async def close(self) -> None:
        """Release any held connections or resources."""
        ...
