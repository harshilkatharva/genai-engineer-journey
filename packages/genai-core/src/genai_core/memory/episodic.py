from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from ..core import get_settings
from ..exceptions import MemoryValidationError
from ..models.memory_model import ActionCheck, ActionEpisode, ActionStatus
from ..observability import ObservabilityEvent, get_logger, safe_log
from .protocol import MemoryStorageAdapter

_logger = get_logger("genai_core.memory.episodic")


def canonical_fingerprint(
    action_type: str,
    parameters: dict[str, Any] | None = None,
    target: str | None = None,
) -> str:
    """Create a deterministic SHA-256 fingerprint from canonical JSON parameters."""
    if not action_type or not action_type.strip():
        raise MemoryValidationError("action_type cannot be empty")
    try:
        payload: dict[str, Any] = {
            "action_type": action_type.strip().casefold(),
            "parameters": parameters or {},
        }
        if target is not None:
            payload["target"] = target.strip().casefold()
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise MemoryValidationError("Action parameters must be JSON-serializable") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class EpisodicMemory:
    """Manages timestamped action episodes, parameter canonicalization, and idempotency checks."""

    def __init__(
        self,
        store: MemoryStorageAdapter,
        retention_days: int | None = None,
    ) -> None:
        self.store = store
        settings = get_settings()
        self.retention_days = (
            retention_days
            if retention_days is not None
            else settings.memory_episodic_retention_days
        )

    async def record(
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
        """Record an attempted or executed action with its canonical parameter hash."""
        if not user_id or not user_id.strip():
            raise MemoryValidationError("user_id is required")
        if not action_type or not action_type.strip():
            raise MemoryValidationError("action_type cannot be empty")

        params = parameters or {}
        param_hash = canonical_fingerprint(action_type, params, target)
        meta = metadata.copy() if metadata else {}
        if target is not None:
            meta["target"] = target

        episode = ActionEpisode(
            user_id=user_id.strip(),
            session_id=session_id.strip() if session_id else None,
            action_type=action_type.strip(),
            parameters=params,
            parameter_hash=param_hash,
            result_summary=result_summary,
            status=status,
            metadata=meta,
        )

        saved = await self.store.record_episode(episode)
        safe_log(
            _logger,
            logging.INFO,
            "Action episode recorded",
            event=ObservabilityEvent.MEMORY_STORE_SUCCESS,
            user_id=user_id,
            action_type=action_type,
            parameter_hash=param_hash,
            status=str(status),
        )
        return saved

    async def check(
        self,
        user_id: str,
        action_type: str,
        parameters: dict[str, Any] | None = None,
        target: str | None = None,
        session_id: str | None = None,
    ) -> ActionCheck:
        """Check whether an identical action was already executed (loop prevention / idempotency)."""
        if not user_id or not user_id.strip():
            raise MemoryValidationError("user_id is required")
        if not action_type or not action_type.strip():
            raise MemoryValidationError("action_type cannot be empty")

        param_hash = canonical_fingerprint(action_type, parameters, target)
        match = await self.store.find_matching_episode(
            user_id=user_id.strip(),
            action_type=action_type.strip(),
            parameter_hash=param_hash,
            session_id=session_id.strip() if session_id else None,
        )

        if match is not None:
            decision = ActionCheck(
                status="do_not_repeat",
                reason=f"Action '{action_type}' with identical parameters was already executed.",
                previous_episode=match,
            )
        else:
            decision = ActionCheck(
                status="execute",
                reason="No prior execution found.",
                previous_episode=None,
            )

        safe_log(
            _logger,
            logging.INFO,
            "Episodic action check evaluated",
            event=ObservabilityEvent.MEMORY_ACTION_CHECK,
            user_id=user_id,
            action_type=action_type,
            parameter_hash=param_hash,
            status=decision.status,
        )
        return decision

    async def get_episodes(
        self,
        user_id: str,
        action_type: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[ActionEpisode]:
        """Retrieve recent action episodes for a user."""
        if not user_id or not user_id.strip():
            raise MemoryValidationError("user_id is required")
        return await self.store.get_episodes(
            user_id=user_id.strip(),
            action_type=action_type,
            session_id=session_id,
            limit=limit,
        )

    async def delete_all(self, user_id: str) -> int:
        """Delete all episodes for a user."""
        if not user_id or not user_id.strip():
            raise MemoryValidationError("user_id is required")
        count = await self.store.delete_episodes(user_id=user_id.strip())
        safe_log(
            _logger,
            logging.INFO,
            "Episodic records deleted for user",
            event=ObservabilityEvent.MEMORY_DELETE_SUCCESS,
            user_id=user_id,
            count=count,
        )
        return count

    async def purge(self, older_than: datetime | None = None, user_id: str | None = None) -> int:
        """Purge episodes older than a given date or default retention window."""
        cutoff = (
            older_than
            if older_than is not None
            else (datetime.now(UTC) - timedelta(days=self.retention_days))
        )
        count = await self.store.purge_episodes(
            older_than=cutoff, user_id=user_id.strip() if user_id else None
        )
        safe_log(
            _logger,
            logging.INFO,
            "Aged episodic records purged",
            event=ObservabilityEvent.MEMORY_PURGE_SUCCESS,
            purged_count=count,
            cutoff=cutoff.isoformat(),
        )
        return count
