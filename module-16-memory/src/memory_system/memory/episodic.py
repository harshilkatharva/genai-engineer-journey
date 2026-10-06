from __future__ import annotations

import hashlib
import json
from datetime import timedelta

from ..core import Settings, get_settings
from ..db.repository import MemoryRepository
from ..models import ActionCheck, ActionEpisode, ActionStatus, utc_now


class EpisodicMemory:
    # Configures persistent action history with the selected retention settings.
    def __init__(self, repository: MemoryRepository, settings: Settings | None = None) -> None:
        self.repository = repository
        self.settings = settings or get_settings()

    # Records an attempted or completed action with its automatic expiration time.
    async def record(
        self,
        user_id: str,
        task_id: str,
        action_type: str,
        target: str,
        parameters: dict[str, object] | None = None,
        status: ActionStatus = "attempted",
        conversation_id: str | None = None,
    ) -> ActionEpisode:
        if not user_id or not task_id or not action_type.strip() or not target.strip():
            raise ValueError("user_id, task_id, action_type, and target are required")
        parameter_values = parameters or {}
        fingerprint = self._fingerprint(action_type, target, parameter_values)
        occurred_at = utc_now()
        episode = ActionEpisode(
            user_id=user_id,
            task_id=task_id,
            conversation_id=conversation_id,
            action_type=action_type,
            target=target,
            parameters=parameter_values,
            status=status,
            occurred_at=occurred_at,
            expires_at=occurred_at + timedelta(days=self.settings.memory_episodic_retention_days),
        )
        return await self.repository.record_episode(episode, fingerprint)

    # Checks whether an equivalent action was already recorded for the task.
    async def check(
        self,
        user_id: str,
        task_id: str,
        action_type: str,
        target: str,
        parameters: dict[str, object] | None = None,
    ) -> ActionCheck:
        if not user_id or not task_id or not action_type.strip() or not target.strip():
            raise ValueError("user_id, task_id, action_type, and target are required")
        fingerprint = self._fingerprint(action_type, target, parameters or {})
        episode = await self.repository.find_episode(user_id, task_id, fingerprint)
        attempted = episode is not None
        return ActionCheck(
            already_attempted=attempted,
            recommendation="do_not_repeat" if attempted else "execute",
            episode=episode,
        )

    # Removes all episodic history belonging to a user.
    async def delete_all(self, user_id: str) -> int:
        if not user_id:
            raise ValueError("user_id is required")
        return await self.repository.delete_episodes(user_id)

    # Creates a stable digest for matching equivalent actions and parameters.
    @staticmethod
    def _fingerprint(action_type: str, target: str, parameters: dict[str, object]) -> str:
        try:
            canonical = json.dumps(
                {
                    "action_type": action_type.strip().casefold(),
                    "target": target.strip().casefold(),
                    "parameters": parameters,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Action parameters must be JSON-serializable") from exc
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
