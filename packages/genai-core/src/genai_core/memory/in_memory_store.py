from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime

from ..models.memory_model import (
    ActionEpisode,
    ConversationMessage,
    LongTermMemoryRecord,
    MemoryCategory,
    MemorySearchResult,
    PinnedGoal,
    WorkingContext,
)
from .protocol import MemoryStorageAdapter, TokenCounter


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot_product / (norm_a * norm_b)))


class InMemoryMemoryStore(MemoryStorageAdapter):
    """In-memory implementation of MemoryStorageAdapter.

    Maintains short-term, long-term, and episodic memories in memory structures with
    strict user and session isolation. Ideal for unit tests, local CLI runs, and ephemeral agents.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        # Short-term: (user_id, session_id) -> data
        self._messages: dict[tuple[str, str], list[ConversationMessage]] = {}
        self._pinned_goals: dict[tuple[str, str], list[PinnedGoal]] = {}
        self._last_accessed: dict[tuple[str, str], datetime] = {}

        # Long-term: user_id -> {record_id -> LongTermMemoryRecord}
        self._records: dict[str, dict[str, LongTermMemoryRecord]] = {}

        # Episodic: user_id -> list[ActionEpisode]
        self._episodes: dict[str, list[ActionEpisode]] = {}

    async def initialize(self) -> None:
        """No-op initialization for in-memory store."""

    async def close(self) -> None:
        """Clear all in-memory structures."""
        async with self._lock:
            self._messages.clear()
            self._pinned_goals.clear()
            self._last_accessed.clear()
            self._records.clear()
            self._episodes.clear()

    # --- Short-Term Operations ---

    async def append_message(
        self, user_id: str, session_id: str, message: ConversationMessage
    ) -> None:
        key = (user_id, session_id)
        now = datetime.now(UTC)
        async with self._lock:
            if key not in self._messages:
                self._messages[key] = []
            self._messages[key].append(message)
            self._last_accessed[key] = now

    async def get_messages(
        self, user_id: str, session_id: str, limit: int | None = None
    ) -> list[ConversationMessage]:
        key = (user_id, session_id)
        async with self._lock:
            messages = self._messages.get(key, []).copy()
            self._last_accessed[key] = datetime.now(UTC)
        if limit is not None and limit > 0:
            return messages[-limit:]
        return messages

    async def clear_messages(self, user_id: str, session_id: str) -> int:
        key = (user_id, session_id)
        async with self._lock:
            messages = self._messages.pop(key, [])
            self._last_accessed[key] = datetime.now(UTC)
            return len(messages)

    async def set_pinned_goals(
        self, user_id: str, session_id: str, goals: list[PinnedGoal]
    ) -> None:
        key = (user_id, session_id)
        async with self._lock:
            self._pinned_goals[key] = goals.copy()
            self._last_accessed[key] = datetime.now(UTC)

    async def get_pinned_goals(self, user_id: str, session_id: str) -> list[PinnedGoal]:
        key = (user_id, session_id)
        async with self._lock:
            goals = self._pinned_goals.get(key, []).copy()
            self._last_accessed[key] = datetime.now(UTC)
            return goals

    async def prune_session(
        self, user_id: str, session_id: str, max_tokens: int, token_counter: TokenCounter
    ) -> WorkingContext:
        key = (user_id, session_id)
        async with self._lock:
            messages = self._messages.get(key, []).copy()
            goals = self._pinned_goals.get(key, []).copy()
            self._last_accessed[key] = datetime.now(UTC)

        # 1. Calculate pinned cost
        pinned_cost = sum(
            token_counter.count(f"Pinned instruction: {goal.goal}\n") for goal in goals
        )
        remaining = max_tokens - pinned_cost

        # 2. Retain messages from newest to oldest within remaining budget
        retained_reversed: list[ConversationMessage] = []
        message_cost = 0
        truncated = False

        if remaining > 0:
            for msg in reversed(messages):
                cost = msg.token_count or token_counter.count_message(msg)
                if cost <= remaining:
                    retained_reversed.append(msg)
                    remaining -= cost
                    message_cost += cost
                else:
                    truncated = True
                    break
        elif messages:
            truncated = True

        retained = list(reversed(retained_reversed))
        if len(retained) < len(messages):
            truncated = True

        total_tokens = pinned_cost + message_cost
        return WorkingContext(
            messages=retained,
            pinned_goals=goals,
            total_tokens=total_tokens,
            truncated=truncated,
        )

    async def clear_session(self, user_id: str, session_id: str) -> None:
        key = (user_id, session_id)
        async with self._lock:
            self._messages.pop(key, None)
            self._pinned_goals.pop(key, None)
            self._last_accessed.pop(key, None)

    async def is_session_expired(
        self, user_id: str, session_id: str, max_age_seconds: float
    ) -> bool:
        key = (user_id, session_id)
        now = datetime.now(UTC)
        async with self._lock:
            last = self._last_accessed.get(key)
            if last is None:
                return True
            return (now - last).total_seconds() >= max_age_seconds

    # --- Long-Term Operations ---

    async def save_record(self, record: LongTermMemoryRecord) -> LongTermMemoryRecord:
        async with self._lock:
            user_records = self._records.setdefault(record.user_id, {})
            # Mark superseded records
            if record.supersedes_ids:
                for sup_id in record.supersedes_ids:
                    if sup_id in user_records:
                        target = user_records[sup_id]
                        user_records[sup_id] = target.model_copy(
                            update={"is_superseded": True, "updated_at": datetime.now(UTC)}
                        )
            user_records[record.id] = record
            return record

    async def get_record(self, user_id: str, record_id: str) -> LongTermMemoryRecord | None:
        async with self._lock:
            return self._records.get(user_id, {}).get(record_id)

    async def list_records(
        self,
        user_id: str,
        category: MemoryCategory | str | None = None,
        include_superseded: bool = False,
        include_expired: bool = False,
        limit: int = 100,
    ) -> list[LongTermMemoryRecord]:
        now = datetime.now(UTC)
        async with self._lock:
            user_records = list(self._records.get(user_id, {}).values())

        filtered: list[LongTermMemoryRecord] = []
        for rec in user_records:
            if not include_superseded and rec.is_superseded:
                continue
            if not include_expired and rec.expires_at is not None and rec.expires_at <= now:
                continue
            if category is not None and rec.category != category:
                continue
            filtered.append(rec)

        filtered.sort(key=lambda r: r.created_at, reverse=True)
        return filtered[:limit]

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
        now = datetime.now(UTC)
        async with self._lock:
            user_records = list(self._records.get(user_id, {}).values())

        # Determine supersession across all records
        all_superseded_ids: set[str] = set()
        for rec in user_records:
            if rec.supersedes_ids and (rec.expires_at is None or rec.expires_at > now):
                all_superseded_ids.update(rec.supersedes_ids)

        results: list[MemorySearchResult] = []
        for rec in user_records:
            # Check expiration
            if rec.expires_at is not None and rec.expires_at <= now:
                continue
            # Check category filter
            if category is not None and rec.category != category:
                continue

            is_superseded = rec.is_superseded or (rec.id in all_superseded_ids)
            if not include_superseded and is_superseded:
                continue

            if not rec.embedding:
                continue

            similarity = _cosine_similarity(query_embedding, rec.embedding)
            if similarity < similarity_threshold:
                continue

            # Exponential recency decay
            delta_days = max(0.0, (now - rec.created_at).total_seconds() / 86400.0)
            decay_rate = math.log(2) / max(1.0, float(half_life_days))
            recency_score = math.exp(-decay_rate * delta_days)

            # Combined ranking score
            rank_score = (0.65 * similarity) + (0.35 * recency_score)
            if is_superseded:
                rank_score -= 2.0

            results.append(
                MemorySearchResult(
                    record=rec.model_copy(update={"is_superseded": is_superseded}),
                    similarity=round(similarity, 4),
                    recency_score=round(recency_score, 4),
                    rank_score=round(rank_score, 4),
                )
            )

        # Sort by rank_score descending, then created_at descending
        results.sort(key=lambda res: (res.rank_score, res.record.created_at), reverse=True)
        return results[:top_k]

    async def supersede_records(self, user_id: str, superseded_ids: list[str]) -> int:
        if not superseded_ids:
            return 0
        now = datetime.now(UTC)
        count = 0
        async with self._lock:
            user_records = self._records.get(user_id, {})
            for rec_id in superseded_ids:
                if rec_id in user_records and not user_records[rec_id].is_superseded:
                    user_records[rec_id] = user_records[rec_id].model_copy(
                        update={"is_superseded": True, "updated_at": now}
                    )
                    count += 1
        return count

    async def delete_record(self, user_id: str, record_id: str) -> bool:
        async with self._lock:
            user_records = self._records.get(user_id, {})
            if record_id in user_records:
                del user_records[record_id]
                return True
            return False

    async def purge_expired(self, user_id: str | None = None) -> int:
        now = datetime.now(UTC)
        purged = 0
        async with self._lock:
            users = [user_id] if user_id else list(self._records.keys())
            for uid in users:
                user_records = self._records.get(uid, {})
                to_delete = [
                    rec_id
                    for rec_id, rec in user_records.items()
                    if rec.expires_at is not None and rec.expires_at <= now
                ]
                for rec_id in to_delete:
                    del user_records[rec_id]
                    purged += 1
        return purged

    # --- Episodic Operations ---

    async def record_episode(self, episode: ActionEpisode) -> ActionEpisode:
        async with self._lock:
            user_episodes = self._episodes.setdefault(episode.user_id, [])
            user_episodes.append(episode)
            return episode

    async def get_episodes(
        self,
        user_id: str,
        action_type: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[ActionEpisode]:
        async with self._lock:
            episodes = self._episodes.get(user_id, []).copy()

        filtered: list[ActionEpisode] = []
        for ep in reversed(episodes):
            if action_type is not None and ep.action_type != action_type:
                continue
            if session_id is not None and ep.session_id != session_id:
                continue
            filtered.append(ep)
            if len(filtered) >= limit:
                break
        return filtered

    async def find_matching_episode(
        self,
        user_id: str,
        action_type: str,
        parameter_hash: str,
        session_id: str | None = None,
    ) -> ActionEpisode | None:
        async with self._lock:
            episodes = self._episodes.get(user_id, [])
            for ep in reversed(episodes):
                if (
                    ep.action_type == action_type
                    and ep.parameter_hash == parameter_hash
                    and (session_id is None or ep.session_id == session_id)
                ):
                    return ep
            return None

    async def delete_episodes(self, user_id: str) -> int:
        async with self._lock:
            episodes = self._episodes.pop(user_id, [])
            return len(episodes)

    async def purge_episodes(self, older_than: datetime, user_id: str | None = None) -> int:
        purged = 0
        async with self._lock:
            users = [user_id] if user_id else list(self._episodes.keys())
            for uid in users:
                episodes = self._episodes.get(uid, [])
                retained = [ep for ep in episodes if ep.occurred_at >= older_than]
                purged += len(episodes) - len(retained)
                self._episodes[uid] = retained
        return purged
