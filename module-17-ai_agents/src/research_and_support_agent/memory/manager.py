"""Memory management layer coordinating genai-core MemoryService with agent lifecycle."""

from __future__ import annotations

import logging
from typing import Any

from genai_core.models import (
    ActionCheck,
    ActionEpisode,
    ActionStatus,
    LongTermMemoryRecord,
    MemoryCategory,
    MemorySearchResult,
)
from genai_core.observability import ObservabilityEvent, get_logger, safe_log
from genai_core.services import MemoryService

_logger = get_logger("research_and_support_agent.memory.manager")


class AgentMemoryManager:
    """Coordinates persistent memory operations, context injection, and episodic safeguards."""

    def __init__(self, memory_service: MemoryService) -> None:
        self.memory_service = memory_service

    # --- Pre-Turn Context Assembly ---

    async def get_context_for_prompt(
        self,
        user_id: str,
        goal: str,
        top_k: int = 3,
    ) -> str:
        """Recall top relevant long-term memories and format them for system prompt injection."""
        if not user_id or not user_id.strip():
            return ""

        try:
            memories: list[MemorySearchResult] = await self.memory_service.recall(
                user_id=user_id.strip(),
                query=goal,
                top_k=top_k,
            )
            if not memories:
                return ""

            lines = ["\n[Recalled Long-Term User Facts & Preferences]:"]
            for m in memories:
                category_tag = f"({m.record.category})" if m.record.category else ""
                lines.append(f"- {m.record.content} {category_tag} [relevance: {m.rank_score:.2f}]")
            return "\n".join(lines) + "\n"
        except Exception as exc:  # noqa: BLE001
            safe_log(
                _logger,
                logging.WARNING,
                f"Memory recall failed during context assembly: {exc}",
                event=ObservabilityEvent.TOOL_EXECUTION_FAILED,
            )
            return ""

    # --- Long-Term Memory Operations ---

    async def get_longterm_memory(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant long-term memories for authorized user."""
        results = await self.memory_service.recall(
            user_id=user_id,
            query=query,
            top_k=top_k,
        )
        return [
            {
                "record_id": r.record.id,
                "content": r.record.content,
                "category": r.record.category,
                "confidence": r.record.confidence,
                "score": round(r.rank_score, 4),
                "created_at": r.record.created_at.isoformat() if r.record.created_at else None,
            }
            for r in results
        ]

    async def save_longterm_memory(
        self,
        user_id: str,
        content: str,
        category: str = "fact",
    ) -> dict[str, Any]:
        """Save a new long-term memory record."""
        cat_enum = (
            MemoryCategory(category)
            if category in MemoryCategory.__members__.values()
            else MemoryCategory.FACT
        )
        record = await self.memory_service.remember(
            user_id=user_id,
            content=content,
            category=cat_enum,
        )
        return {
            "record_id": record.id,
            "user_id": record.user_id,
            "content": record.content,
            "category": record.category,
            "status": "saved",
        }

    async def delete_longterm_all(self, user_id: str) -> dict[str, Any]:
        """Delete all long-term memory records for the authorized user."""
        store = self.memory_service.store
        records: list[LongTermMemoryRecord] = await store.list_records(
            user_id=user_id,
            include_superseded=True,
            include_expired=True,
            limit=500,
        )
        deleted_count = 0
        for r in records:
            success = await self.memory_service.long_term.delete(user_id=user_id, record_id=r.id)
            if success:
                deleted_count += 1

        return {
            "user_id": user_id,
            "deleted_count": deleted_count,
            "status": "all_longterm_deleted",
        }

    async def delete_longterm_specific(self, user_id: str, record_id: str) -> dict[str, Any]:
        """Delete a specific long-term memory record by ID."""
        success = await self.memory_service.long_term.delete(
            user_id=user_id,
            record_id=record_id,
        )
        return {
            "user_id": user_id,
            "record_id": record_id,
            "deleted": success,
            "status": "deleted" if success else "not_found",
        }

    # --- Episodic Memory Operations ---

    async def get_episodic_memory(
        self,
        user_id: str,
        action_type: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Retrieve recent episodic actions and outcomes for the user."""
        episodes: list[ActionEpisode] = await self.memory_service.episodic.get_episodes(
            user_id=user_id,
            action_type=action_type,
            limit=limit,
        )
        return [
            {
                "episode_id": ep.episode_id,
                "action_type": ep.action_type,
                "parameters": ep.parameters,
                "parameter_hash": ep.parameter_hash,
                "result_summary": ep.result_summary,
                "status": ep.status,
                "created_at": ep.created_at.isoformat() if ep.created_at else None,
            }
            for ep in episodes
        ]

    async def delete_episodic_all(self, user_id: str) -> dict[str, Any]:
        """Delete all episodic action history for the authorized user."""
        deleted_count = await self.memory_service.episodic.delete_all(user_id=user_id)
        return {
            "user_id": user_id,
            "deleted_count": deleted_count,
            "status": "all_episodes_deleted",
        }

    async def delete_episodic_specific(
        self,
        user_id: str,
        episode_ids: list[str],
    ) -> dict[str, Any]:
        """Delete selected episodic records by ID for authorized user."""
        store = self.memory_service.store
        deleted_count = 0

        # If PostgreSQL store is active and has raw table access, delete selected IDs directly
        if hasattr(store, "_connection_context") and hasattr(store, "episodic_table"):
            try:
                async with store._connection_context() as conn:
                    if conn is not None:
                        async with conn.cursor() as cur:
                            await cur.execute(
                                f"DELETE FROM {store.episodic_table} WHERE user_id = %s AND id = ANY(%s)",
                                (user_id, episode_ids),
                            )
                            deleted_count = cur.rowcount
                        if hasattr(conn, "commit"):
                            await conn.commit()
            except Exception as exc:  # noqa: BLE001
                safe_log(_logger, logging.WARNING, f"Postgres episode deletion error: {exc}")

        # In-memory store fallback: filter out target episode IDs
        elif hasattr(store, "_episodes"):
            async with store._lock:
                current = store._episodes.get(user_id, [])
                initial_len = len(current)
                store._episodes[user_id] = [
                    ep for ep in current if ep.episode_id not in episode_ids
                ]
                deleted_count = initial_len - len(store._episodes[user_id])

        return {
            "user_id": user_id,
            "deleted_count": deleted_count,
            "requested_ids": episode_ids,
            "status": "episodes_deleted",
        }

    async def purge_expired_records(self, user_id: str | None = None) -> dict[str, Any]:
        """Purge only expired memory records."""
        expired_count = await self.memory_service.long_term.purge_expired(user_id=user_id)
        return {
            "user_id": user_id,
            "expired_purged_count": expired_count,
            "status": "expired_purged",
        }

    # --- Episodic Action Checking & Idempotency Safeguards ---

    async def check_action_idempotency(
        self,
        user_id: str,
        action_type: str,
        parameters: dict[str, Any] | None = None,
    ) -> ActionCheck:
        """Check if an identical action was already recorded."""
        return await self.memory_service.check_action(
            user_id=user_id,
            action_type=action_type,
            parameters=parameters,
        )

    async def record_action_outcome(
        self,
        user_id: str,
        action_type: str,
        parameters: dict[str, Any] | None = None,
        result_summary: str | None = None,
        status: ActionStatus | str = ActionStatus.SUCCESS,
    ) -> ActionEpisode:
        """Record the outcome of a tool execution episode."""
        return await self.memory_service.record_action(
            user_id=user_id,
            action_type=action_type,
            parameters=parameters,
            result_summary=result_summary,
            status=status,
        )
