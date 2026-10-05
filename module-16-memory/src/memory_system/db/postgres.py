from __future__ import annotations

import json
from uuid import UUID

import asyncpg
from pgvector.asyncpg import register_vector

from ..models.memory import (
    ActionEpisode,
    LongTermMemoryRecord,
    MemorySearchResult,
)
from .repository import MemoryRepository


class PostgresMemoryStore(MemoryRepository):
    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    @classmethod
    async def connect(
        cls, database_url: str, min_size: int = 1, max_size: int = 10
    ) -> PostgresMemoryStore:
        if not database_url:
            raise ValueError("DATABASE_URL is required")
        pool = await asyncpg.create_pool(
            database_url,
            min_size=min_size,
            max_size=max_size,
            init=register_vector,
        )
        if pool is None:
            raise RuntimeError("PostgreSQL pool initialization failed")
        return cls(pool)

    async def close(self) -> None:
        await self.pool.close()

    async def save_memory(
        self, memory: LongTermMemoryRecord, embedding: list[float]
    ) -> LongTermMemoryRecord:
        async with self.pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO long_term_memories
                    (id, user_id, content, category, supersedes_ids, created_at, expires_at, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                memory.id,
                memory.user_id,
                memory.content,
                memory.category,
                memory.supersedes_ids,
                memory.created_at,
                memory.expires_at,
                embedding,
            )
        return memory

    async def list_memories(self, user_id: str, limit: int) -> list[LongTermMemoryRecord]:
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT id, user_id, content, category, supersedes_ids, created_at, expires_at
                FROM long_term_memories
                WHERE user_id = $1 AND (expires_at IS NULL OR expires_at > now())
                ORDER BY created_at DESC
                LIMIT $2
                """,
                user_id,
                limit,
            )
        return [_memory_from_row(row) for row in rows]

    async def search_memories(
        self, user_id: str, embedding: list[float], limit: int, half_life_days: int
    ) -> list[MemorySearchResult]:
        candidate_limit = min(max(limit * 10, 100), 5_000)
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                """
                WITH candidates AS (
                    SELECT m.*,
                        1 - (m.embedding <=> $2::vector) AS similarity
                    FROM long_term_memories m
                    WHERE m.user_id = $1
                      AND (m.expires_at IS NULL OR m.expires_at > now())
                    ORDER BY m.embedding <=> $2::vector
                    LIMIT $3
                ), scored AS (
                    SELECT candidates.*,
                        EXISTS (
                            SELECT 1
                            FROM long_term_memories newer
                            WHERE newer.user_id = candidates.user_id
                              AND candidates.id = ANY(newer.supersedes_ids)
                              AND (newer.expires_at IS NULL OR newer.expires_at > now())
                        ) AS is_superseded
                    FROM candidates
                )
                SELECT *,
                    (0.65 * similarity)
                    + (0.35 * exp(-ln(2) * GREATEST(
                        0, extract(epoch FROM (now() - created_at)) / 86400
                    ) / $4))
                    - CASE WHEN is_superseded THEN 2.0 ELSE 0 END AS rank_score
                FROM scored
                ORDER BY rank_score DESC, created_at DESC
                LIMIT $5
                """,
                user_id,
                embedding,
                candidate_limit,
                half_life_days,
                limit,
            )
        return [
            MemorySearchResult(
                memory=_memory_from_row(row),
                similarity=float(row["similarity"]),
                rank_score=float(row["rank_score"]),
                superseded=bool(row["is_superseded"]),
            )
            for row in rows
        ]

    async def delete_memory(self, user_id: str, memory_id: UUID) -> bool:
        async with self.pool.acquire() as connection:
            status = await connection.execute(
                "DELETE FROM long_term_memories WHERE user_id = $1 AND id = $2",
                user_id,
                memory_id,
            )
        return status.endswith("1")

    async def delete_all_user_data(self, user_id: str) -> tuple[int, int]:
        async with self.pool.acquire() as connection, connection.transaction():
            memories = await connection.fetchval(
                "WITH deleted AS (DELETE FROM long_term_memories WHERE user_id = $1 RETURNING 1) "
                "SELECT count(*) FROM deleted",
                user_id,
            )
            episodes = await connection.fetchval(
                "WITH deleted AS (DELETE FROM episodic_memories WHERE user_id = $1 RETURNING 1) "
                "SELECT count(*) FROM deleted",
                user_id,
            )
        return int(memories), int(episodes)

    async def purge_expired(self) -> tuple[int, int]:
        async with self.pool.acquire() as connection, connection.transaction():
            memories = await connection.fetchval(
                "WITH deleted AS (DELETE FROM long_term_memories "
                "WHERE expires_at <= now() RETURNING 1) SELECT count(*) FROM deleted"
            )
            episodes = await connection.fetchval(
                "WITH deleted AS (DELETE FROM episodic_memories "
                "WHERE expires_at <= now() RETURNING 1) SELECT count(*) FROM deleted"
            )
        return int(memories), int(episodes)

    async def record_episode(self, episode: ActionEpisode, fingerprint: str) -> ActionEpisode:
        try:
            parameters_json = json.dumps(episode.parameters, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("Action parameters must be JSON-serializable") from exc
        async with self.pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO episodic_memories
                    (id, user_id, task_id, conversation_id, action_type, target,
                     parameters, action_fingerprint, status, occurred_at, expires_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11)
                """,
                episode.id,
                episode.user_id,
                episode.task_id,
                episode.conversation_id,
                episode.action_type,
                episode.target,
                parameters_json,
                fingerprint,
                episode.status,
                episode.occurred_at,
                episode.expires_at,
            )
        return episode

    async def find_episode(
        self, user_id: str, task_id: str, fingerprint: str
    ) -> ActionEpisode | None:
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                SELECT id, user_id, task_id, conversation_id, action_type, target,
                       parameters, status, occurred_at, expires_at
                FROM episodic_memories
                WHERE user_id = $1 AND task_id = $2 AND action_fingerprint = $3
                  AND expires_at > now()
                ORDER BY occurred_at DESC
                LIMIT 1
                """,
                user_id,
                task_id,
                fingerprint,
            )
        return _episode_from_row(row) if row else None

    async def delete_episodes(self, user_id: str) -> int:
        async with self.pool.acquire() as connection:
            status = await connection.execute(
                "DELETE FROM episodic_memories WHERE user_id = $1", user_id
            )
        return int(status.rsplit(" ", 1)[-1])


def _memory_from_row(row: asyncpg.Record) -> LongTermMemoryRecord:
    return LongTermMemoryRecord(
        id=row["id"],
        user_id=row["user_id"],
        content=row["content"],
        category=row["category"],
        supersedes_ids=row["supersedes_ids"],
        created_at=row["created_at"],
        expires_at=row["expires_at"],
    )


def _episode_from_row(row: asyncpg.Record) -> ActionEpisode:
    parameters = row["parameters"]
    if isinstance(parameters, str):
        parameters = json.loads(parameters)
    return ActionEpisode(
        id=row["id"],
        user_id=row["user_id"],
        task_id=row["task_id"],
        conversation_id=row["conversation_id"],
        action_type=row["action_type"],
        target=row["target"],
        parameters=parameters,
        status=row["status"],
        occurred_at=row["occurred_at"],
        expires_at=row["expires_at"],
    )
