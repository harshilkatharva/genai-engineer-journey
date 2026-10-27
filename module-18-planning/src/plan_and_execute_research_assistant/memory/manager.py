"""Memory manager coordinating long-term preferences and episodic plan retrieval."""

from __future__ import annotations

import inspect
import logging
from typing import Any

from genai_core import (
    ActionEpisode,
    ActionStatus,
    MemoryService,
    ObservabilityEvent,
    get_logger,
    safe_log,
)

from ..core.config import AssistantConfig, get_assistant_config
from ..models.plan import Plan

_logger = get_logger("plan_and_execute_research_assistant.memory")


class PlanningMemoryManager:
    """Manages memory interactions for planning, recalling preferences and past plan episodes."""

    def __init__(
        self,
        memory_service: MemoryService,
        config: AssistantConfig | None = None,
    ) -> None:
        self.memory_service = memory_service
        self.config = config or get_assistant_config()

    async def get_user_preferences(
        self,
        user_id: str,
        query: str,
        top_k: int | None = None,
    ) -> list[str]:
        """Recall relevant long-term user preferences and domain constraints.

        Args:
            user_id: Authorized user ID.
            query: Goal or query string to match semantically against preferences.
            top_k: Maximum number of records to retrieve (defaults to config).

        Returns:
            List of recalled preference text strings.
        """
        if not user_id or not user_id.strip():
            return []

        limit = top_k if top_k is not None else self.config.max_recalled_preferences
        try:
            results = await self.memory_service.search_longterm(
                user_id=user_id.strip(),
                query=query.strip(),
                limit=limit,
            )
            preferences: list[str] = []
            for r in results:
                # Handle MemorySearchResult where content lives on r.record.content
                if hasattr(r, "record") and hasattr(r.record, "content"):
                    text = r.record.content
                else:
                    text = getattr(r, "content", str(r))
                if text and text.strip():
                    preferences.append(text.strip())

            safe_log(
                _logger,
                logging.INFO,
                f"Recalled {len(preferences)} user preferences for user '{user_id}'",
                event=ObservabilityEvent.MEMORY_RETRIEVE_SUCCESS,
                user_id=user_id,
                count=len(preferences),
            )
            return preferences
        except Exception as exc:  # noqa: BLE001
            safe_log(
                _logger,
                logging.WARNING,
                f"Failed to recall preferences for user '{user_id}': {exc}",
                user_id=user_id,
                error=str(exc),
            )
            return []

    async def get_similar_past_plans(
        self,
        user_id: str,
        goal: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve similar past plan episodes if enabled by configuration.

        Operates on a config-based toggle (AssistantConfig.enable_episodic_similarity_search).
        When False (default until genai-core natively provides episodic vector search),
        bypasses episodic retrieval to save tokens and latency. When True, queries episodic
        similarity search or falls back to recent plan execution episodes.

        Args:
            user_id: Authorized user ID.
            goal: Goal text to match.
            limit: Maximum episodes to retrieve.

        Returns:
            List of past plan summaries with keys: goal, result_summary, and steps.
        """
        if not user_id or not user_id.strip():
            return []

        # Config-based gate: return empty if disabled
        if not self.config.enable_episodic_similarity_search:
            safe_log(
                _logger,
                logging.DEBUG,
                "Episodic similarity search disabled via config; skipping past plans recall.",
                user_id=user_id,
            )
            return []

        effective_limit = limit if limit is not None else self.config.max_recalled_past_plans

        try:
            episodes: list[ActionEpisode] = []
            # Check for planned semantic search method on MemoryService or EpisodicMemory
            search_fn = getattr(self.memory_service, "search_episodes", None)
            if search_fn is None:
                search_fn = getattr(self.memory_service.episodic, "search", None)

            # Only invoke if search_fn is an actual async callable
            if callable(search_fn) and inspect.iscoroutinefunction(search_fn):
                episodes = await search_fn(
                    user_id=user_id.strip(),
                    query=goal.strip(),
                    limit=effective_limit,
                )
            else:
                # Fallback: retrieve recent plan_execution episodes
                episodes = await self.memory_service.episodic.get_episodes(
                    user_id=user_id.strip(),
                    action_type="plan_execution",
                    limit=effective_limit,
                )

            past_plans: list[dict[str, Any]] = []
            for ep in episodes:
                params = ep.parameters or {}
                meta = ep.metadata or {}
                past_plans.append(
                    {
                        "goal": params.get("goal") or getattr(ep, "target", None) or "Unknown goal",
                        "result_summary": ep.result_summary or "Completed successfully",
                        "steps": meta.get("step_descriptions", []),
                    }
                )

            safe_log(
                _logger,
                logging.INFO,
                f"Recalled {len(past_plans)} past plan episodes for user '{user_id}'",
                event=ObservabilityEvent.MEMORY_RETRIEVE_SUCCESS,
                user_id=user_id,
                count=len(past_plans),
            )
            return past_plans

        except Exception as exc:  # noqa: BLE001
            safe_log(
                _logger,
                logging.WARNING,
                f"Failed to recall past plan episodes for user '{user_id}': {exc}",
                user_id=user_id,
                error=str(exc),
            )
            return []

    async def record_plan_episode(
        self,
        user_id: str,
        goal: str,
        plan: Plan,
        result_summary: str | None = None,
        status: ActionStatus | str = ActionStatus.SUCCESS,
        metadata: dict[str, Any] | None = None,
    ) -> ActionEpisode:
        """Record a completed or finalized plan execution into episodic memory.

        Args:
            user_id: Authorized user ID.
            goal: Original goal executed.
            plan: Final Plan instance.
            result_summary: Consolidated final deliverable or outcome summary.
            status: Action execution status.
            metadata: Additional telemetry or metadata tags.

        Returns:
            Persisted ActionEpisode instance.
        """
        step_descriptions = [f"[{s.id}]: {s.description}" for s in plan.steps]
        meta: dict[str, Any] = {
            **(metadata or {}),
            "step_count": len(plan.steps),
            "plan_version": plan.version,
            "step_descriptions": step_descriptions,
        }

        episode = await self.memory_service.episodic.record(
            user_id=user_id.strip(),
            action_type="plan_execution",
            parameters={
                "goal": goal.strip(),
                "plan_version": plan.version,
                "step_count": len(plan.steps),
            },
            target="planner",
            result_summary=result_summary,
            status=status,
            metadata=meta,
        )
        safe_log(
            _logger,
            logging.INFO,
            f"Recorded plan execution episode for user '{user_id}' (version={plan.version})",
            event=ObservabilityEvent.MEMORY_STORE_SUCCESS,
            user_id=user_id,
            action_type="plan_execution",
        )
        return episode
