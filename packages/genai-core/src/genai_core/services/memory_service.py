from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..core import Settings, get_settings
from ..exceptions import MemoryValidationError
from ..memory.episodic import EpisodicMemory
from ..memory.in_memory_store import InMemoryMemoryStore
from ..memory.long_term import LongTermMemory
from ..memory.protocol import MemoryStorageAdapter, TokenCounter
from ..memory.short_term import ShortTermMemory
from ..models.memory_model import (
    ActionCheck,
    ActionEpisode,
    ActionStatus,
    ConsolidationResult,
    ConversationMessage,
    LongTermMemoryRecord,
    MemoryCandidate,
    MemoryCategory,
    MemorySearchResult,
    MessageRole,
    PinnedGoal,
    WorkingContext,
)

if TYPE_CHECKING:
    from ..services.embedding_service import EmbeddingService
    from ..services.llm_services import LLMService


class MemoryService:
    """Unified orchestration service for short-term, long-term, and episodic memory.

    Exposes subsystem facades and convenience methods with caller-selectable storage adapters.
    """

    def __init__(
        self,
        store: MemoryStorageAdapter | None = None,
        embedding_service: EmbeddingService | None = None,
        llm_service: LLMService | None = None,
        token_counter: TokenCounter | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store: MemoryStorageAdapter = store if store is not None else InMemoryMemoryStore()
        self.embedding_service = embedding_service
        self.llm_service = llm_service

        self.short_term = ShortTermMemory(
            store=self.store,
            token_counter=token_counter,
            ttl_hours=self.settings.memory_working_ttl_hours,
        )
        self.long_term = LongTermMemory(
            store=self.store,
            embedding_service=self.embedding_service,
            llm_service=self.llm_service,
            recency_half_life_days=self.settings.memory_recency_half_life_days,
        )
        self.episodic = EpisodicMemory(
            store=self.store,
            retention_days=self.settings.memory_episodic_retention_days,
        )

    async def initialize(self) -> None:
        """Initialize underlying storage resources or schemas."""
        await self.store.initialize()

    async def close(self) -> None:
        """Release underlying storage resources."""
        await self.store.close()

    # --- Short-Term Convenience APIs ---

    async def add_message(
        self,
        user_id: str,
        session_id: str,
        role: MessageRole | str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationMessage:
        """Add a conversation message to the session's working memory."""
        return await self.short_term.add_message(
            user_id=user_id,
            session_id=session_id,
            role=role,
            content=content,
            metadata=metadata,
        )

    async def get_messages(
        self,
        user_id: str,
        session_id: str,
        limit: int | None = None,
    ) -> list[ConversationMessage]:
        """Retrieve recent conversation messages."""
        return await self.short_term.get_messages(
            user_id=user_id, session_id=session_id, limit=limit
        )

    async def get_working_context(
        self,
        user_id: str,
        session_id: str,
        token_budget: int | None = None,
    ) -> WorkingContext:
        """Assemble token-budgeted working context snapshot with pinned goals."""
        return await self.short_term.snapshot(
            user_id=user_id,
            session_id=session_id,
            token_budget=token_budget,
        )

    async def pin_goal(
        self,
        user_id: str,
        session_id: str,
        goal: str,
        metadata: dict[str, Any] | None = None,
    ) -> PinnedGoal:
        """Pin a persistent agent goal or instruction."""
        return await self.short_term.pin_goal(
            user_id=user_id,
            session_id=session_id,
            goal=goal,
            metadata=metadata,
        )

    # --- Long-Term Convenience APIs ---

    async def remember(
        self,
        user_id: str,
        content: str | MemoryCandidate,
        category: MemoryCategory | str = MemoryCategory.FACT,
        confidence: float = 1.0,
        supersedes_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LongTermMemoryRecord:
        """Store a persistent fact, preference, or goal into long-term memory."""
        if isinstance(content, MemoryCandidate):
            candidate = content
        else:
            candidate = MemoryCandidate(
                content=content,
                category=category,
                confidence=confidence,
                supersedes_ids=supersedes_ids or [],
                metadata=metadata or {},
            )
        return await self.long_term.store(user_id=user_id, candidate=candidate)

    async def recall(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        category: MemoryCategory | str | None = None,
        similarity_threshold: float = 0.0,
        include_superseded: bool = False,
    ) -> list[MemorySearchResult]:
        """Search user long-term memories ranked by similarity and recency decay."""
        return await self.long_term.search(
            user_id=user_id,
            query=query,
            top_k=top_k,
            category=category,
            similarity_threshold=similarity_threshold,
            include_superseded=include_superseded,
        )

    async def consolidate(
        self,
        user_id: str,
        session_id: str | None = None,
        context: WorkingContext | list[ConversationMessage] | None = None,
        custom_prompt_template: str | None = None,
        llm_service: LLMService | None = None,
        llm_provider: str | None = None,
        similarity_threshold: float = 0.95,
    ) -> ConsolidationResult:
        """Consolidate conversation context into durable long-term memories using LLM extraction.

        Accepts either:
        - A session_id to retrieve turns from short-term memory.
        - A context (WorkingContext or list[ConversationMessage]) directly supplied by the caller.

        Exactly one context source must be provided.
        """
        if not user_id or not user_id.strip():
            raise MemoryValidationError("user_id is required")

        uid = user_id.strip()

        # Validate context source exclusivity
        if session_id is not None and context is not None:
            raise MemoryValidationError(
                "Cannot provide both session_id and context simultaneously; choose exactly one context source"
            )
        if session_id is None and context is None:
            raise MemoryValidationError(
                "Either session_id or context must be provided for memory consolidation"
            )

        messages: list[ConversationMessage]
        resolved_session_id: str | None = None

        if session_id is not None:
            resolved_session_id = session_id.strip()
            messages = await self.short_term.get_messages(
                user_id=uid, session_id=resolved_session_id
            )
        elif isinstance(context, WorkingContext):
            messages = context.messages
        elif isinstance(context, list):
            messages = context
        else:
            raise MemoryValidationError(
                "context must be a WorkingContext or list of ConversationMessage instances"
            )

        return await self.long_term.consolidate(
            user_id=uid,
            messages=messages,
            custom_prompt_template=custom_prompt_template,
            llm_service=llm_service or self.llm_service,
            llm_provider=llm_provider,
            similarity_threshold=similarity_threshold,
            session_id=resolved_session_id,
        )

    # --- Episodic Convenience APIs ---

    async def record_action(
        self,
        user_id: str,
        action_type: str,
        parameters: dict[str, Any] | None = None,
        target: str | None = None,
        session_id: str | None = None,
        result_summary: str | None = None,
        status: ActionStatus | str = ActionStatus.SUCCESS,
        metadata: dict[str, Any] | None = None,
    ) -> ActionEpisode:
        """Record an attempted or completed tool action episode."""
        return await self.episodic.record(
            user_id=user_id,
            action_type=action_type,
            parameters=parameters,
            target=target,
            session_id=session_id,
            result_summary=result_summary,
            status=status,
            metadata=metadata,
        )

    async def check_action(
        self,
        user_id: str,
        action_type: str,
        parameters: dict[str, Any] | None = None,
        target: str | None = None,
        session_id: str | None = None,
    ) -> ActionCheck:
        """Check whether an identical action was already executed."""
        return await self.episodic.check(
            user_id=user_id,
            action_type=action_type,
            parameters=parameters,
            target=target,
            session_id=session_id,
        )

    # --- Maintenance Operations ---

    async def purge_all(self, user_id: str | None = None) -> dict[str, int]:
        """Purge expired long-term records and aged episodic action history."""
        expired_lt = await self.long_term.purge_expired(user_id=user_id)
        aged_ep = await self.episodic.purge(user_id=user_id)
        return {
            "expired_long_term": expired_lt,
            "aged_episodes": aged_ep,
        }
