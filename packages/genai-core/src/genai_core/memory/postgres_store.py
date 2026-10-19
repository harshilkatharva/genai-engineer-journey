from __future__ import annotations

import contextlib
import inspect
import json
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from ..exceptions import MemoryStorageError, MissingExtraDependencyError
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

IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_identifier(name: str, field_name: str) -> str:
    """Validate that a SQL identifier is alphanumeric and safe from SQL injection."""
    if not IDENTIFIER_PATTERN.match(name):
        raise ValueError(
            f"Invalid SQL identifier for {field_name}: '{name}'. "
            "Identifiers must start with a letter or underscore and contain only alphanumeric "
            "characters or underscores."
        )
    return name


def _check_postgres_dependencies() -> None:
    """Verify that optional PostgreSQL and pgvector libraries are installed."""
    try:
        import pgvector  # type: ignore[import-not-found]  # noqa: F401
        import psycopg  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as err:
        raise MissingExtraDependencyError(
            extra="postgres",
            package="psycopg[binary] and pgvector",
            feature="PostgresMemoryStore",
        ) from err


def _format_vector(vec: list[float] | None) -> str | None:
    """Format float list into PostgreSQL vector literal '[x,y,z]'."""
    if vec is None:
        return None
    return f"[{','.join(str(float(x)) for x in vec)}]"


def _to_vector_param(vec: list[float]) -> Any:
    """Wrap float list into pgvector Vector for psycopg parameter binding."""
    try:
        from pgvector import Vector  # type: ignore[import-not-found]

        class _VectorParam(Vector):  # type: ignore[misc]
            def __eq__(self, other: object) -> bool:
                if isinstance(other, (list, tuple)):
                    return self.to_list() == list(other)
                return super().__eq__(other)

        return _VectorParam(vec)
    except Exception:  # noqa: BLE001
        return vec


class PostgresMemoryStore(MemoryStorageAdapter):
    """PostgreSQL + pgvector storage adapter for short-term, long-term, and episodic memory.

    Provides durable, user-isolated persistence with pgvector cosine similarity,
    in-database recency scoring, and parameter-fingerprinted action history.
    """

    def __init__(
        self,
        *,
        dimensions: int = 384,
        connection: Any = None,
        connection_pool: Any = None,
        connection_factory: Callable[[], Any] | None = None,
        connection_string: str | None = None,
        schema: str = "public",
        table_prefix: str = "memory_",
    ) -> None:
        _check_postgres_dependencies()
        self.dimensions = dimensions
        self.schema = _validate_identifier(schema, "schema")
        self.table_prefix = _validate_identifier(table_prefix, "table_prefix")

        self.short_term_messages_table = f"{self.table_prefix}st_messages"
        self.short_term_goals_table = f"{self.table_prefix}st_goals"
        self.short_term_sessions_table = f"{self.table_prefix}st_sessions"
        self.long_term_table = f"{self.table_prefix}lt_records"
        self.episodic_table = f"{self.table_prefix}ep_episodes"

        self._connection = connection
        self._connection_pool = connection_pool
        self._connection_factory = connection_factory
        self._connection_string = connection_string

    @asynccontextmanager
    async def _connection_context(self) -> AsyncIterator[Any]:
        if self._connection_pool is not None:
            async with self._connection_pool.connection() as conn:
                yield conn
        elif self._connection is not None:
            yield self._connection
        elif self._connection_factory is not None:
            conn = self._connection_factory()
            if inspect.isawaitable(conn):
                conn = await conn
            if hasattr(conn, "__aenter__"):
                async with conn as c:
                    yield c
            else:
                yield conn
        elif self._connection_string is not None:
            import psycopg

            async with await psycopg.AsyncConnection.connect(self._connection_string) as conn:
                yield conn
        else:
            raise MemoryStorageError("No PostgreSQL connection or pool configured")

    async def initialize(self) -> None:
        """Create required vector extension, tables, and indexes."""
        stmts = [
            "CREATE EXTENSION IF NOT EXISTS vector;",
            f"""
            CREATE TABLE IF NOT EXISTS {self.short_term_sessions_table} (
                user_id text NOT NULL,
                session_id text NOT NULL,
                last_accessed_at timestamptz NOT NULL DEFAULT now(),
                created_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (user_id, session_id)
            );
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {self.short_term_messages_table} (
                id uuid PRIMARY KEY,
                user_id text NOT NULL,
                session_id text NOT NULL,
                role text NOT NULL,
                content text NOT NULL,
                token_count integer,
                metadata jsonb NOT NULL DEFAULT '{{}}',
                created_at timestamptz NOT NULL DEFAULT now()
            );
            """,
            f"""
            CREATE INDEX IF NOT EXISTS {self.short_term_messages_table}_idx
                ON {self.short_term_messages_table} (user_id, session_id, created_at ASC);
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {self.short_term_goals_table} (
                id uuid PRIMARY KEY,
                user_id text NOT NULL,
                session_id text NOT NULL,
                goal text NOT NULL,
                metadata jsonb NOT NULL DEFAULT '{{}}',
                pinned_at timestamptz NOT NULL DEFAULT now()
            );
            """,
            f"""
            CREATE INDEX IF NOT EXISTS {self.short_term_goals_table}_idx
                ON {self.short_term_goals_table} (user_id, session_id, pinned_at ASC);
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {self.long_term_table} (
                id uuid PRIMARY KEY,
                user_id text NOT NULL,
                content text NOT NULL,
                category text NOT NULL,
                confidence double precision NOT NULL DEFAULT 1.0,
                supersedes_ids uuid[] NOT NULL DEFAULT '{{}}',
                is_superseded boolean NOT NULL DEFAULT false,
                metadata jsonb NOT NULL DEFAULT '{{}}',
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now(),
                expires_at timestamptz,
                embedding vector({self.dimensions})
            );
            """,
            f"""
            CREATE INDEX IF NOT EXISTS {self.long_term_table}_user_idx
                ON {self.long_term_table} (user_id, created_at DESC);
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {self.episodic_table} (
                id uuid PRIMARY KEY,
                user_id text NOT NULL,
                session_id text,
                action_type text NOT NULL,
                parameters jsonb NOT NULL DEFAULT '{{}}',
                parameter_hash text NOT NULL,
                result_summary text,
                status text NOT NULL DEFAULT 'success',
                occurred_at timestamptz NOT NULL DEFAULT now(),
                expires_at timestamptz,
                metadata jsonb NOT NULL DEFAULT '{{}}'
            );
            """,
            f"""
            CREATE INDEX IF NOT EXISTS {self.episodic_table}_lookup_idx
                ON {self.episodic_table} (user_id, action_type, parameter_hash, occurred_at DESC);
            """,
            f"""
            CREATE INDEX IF NOT EXISTS {self.episodic_table}_expiry_idx
                ON {self.episodic_table} (expires_at);
            """,
        ]
        async with self._connection_context() as conn:
            async with conn.cursor() as cur:
                for sql in stmts:
                    await cur.execute(sql)
            if hasattr(conn, "commit"):
                await conn.commit()

    async def close(self) -> None:
        """Close connection pool or connection if owned."""
        if self._connection_pool is not None and hasattr(self._connection_pool, "close"):
            with contextlib.suppress(Exception):
                await self._connection_pool.close()
        elif self._connection is not None and hasattr(self._connection, "close"):
            with contextlib.suppress(Exception):
                await self._connection.close()

    # --- Short-Term Operations ---

    async def append_message(
        self, user_id: str, session_id: str, message: ConversationMessage
    ) -> None:
        async with self._connection_context() as conn:
            async with conn.cursor() as cur:
                # Update session access
                await cur.execute(
                    f"""
                    INSERT INTO {self.short_term_sessions_table} (user_id, session_id, last_accessed_at)
                    VALUES (%s, %s, now())
                    ON CONFLICT (user_id, session_id) DO UPDATE SET last_accessed_at = now()
                    """,
                    (user_id, session_id),
                )
                await cur.execute(
                    f"""
                    INSERT INTO {self.short_term_messages_table}
                    (id, user_id, session_id, role, content, token_count, metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        UUID(message.id),
                        user_id,
                        session_id,
                        str(message.role),
                        message.content,
                        message.token_count,
                        json.dumps(message.metadata),
                        message.timestamp,
                    ),
                )
            if hasattr(conn, "commit"):
                await conn.commit()

    async def get_messages(
        self, user_id: str, session_id: str, limit: int | None = None
    ) -> list[ConversationMessage]:
        async with self._connection_context() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    INSERT INTO {self.short_term_sessions_table} (user_id, session_id, last_accessed_at)
                    VALUES (%s, %s, now())
                    ON CONFLICT (user_id, session_id) DO UPDATE SET last_accessed_at = now()
                    """,
                    (user_id, session_id),
                )
                if limit is not None and limit > 0:
                    query = f"""
                    SELECT id, role, content, token_count, metadata, created_at
                    FROM (
                        SELECT id, role, content, token_count, metadata, created_at
                        FROM {self.short_term_messages_table}
                        WHERE user_id = %s AND session_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    ) sub
                    ORDER BY created_at ASC
                    """
                    await cur.execute(query, (user_id, session_id, limit))
                else:
                    query = f"""
                    SELECT id, role, content, token_count, metadata, created_at
                    FROM {self.short_term_messages_table}
                    WHERE user_id = %s AND session_id = %s
                    ORDER BY created_at ASC
                    """
                    await cur.execute(query, (user_id, session_id))
                rows = await cur.fetchall()
            if hasattr(conn, "commit"):
                await conn.commit()

        return [
            ConversationMessage(
                id=str(row[0]),
                role=row[1],
                content=row[2],
                token_count=row[3],
                metadata=row[4] if isinstance(row[4], dict) else json.loads(row[4] or "{}"),
                timestamp=row[5],
            )
            for row in rows
        ]

    async def clear_messages(self, user_id: str, session_id: str) -> int:
        async with self._connection_context() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"DELETE FROM {self.short_term_messages_table} WHERE user_id = %s AND session_id = %s",
                    (user_id, session_id),
                )
                count = cur.rowcount
            if hasattr(conn, "commit"):
                await conn.commit()
        return max(count, 0)

    async def set_pinned_goals(
        self, user_id: str, session_id: str, goals: list[PinnedGoal]
    ) -> None:
        async with self._connection_context() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"DELETE FROM {self.short_term_goals_table} WHERE user_id = %s AND session_id = %s",
                    (user_id, session_id),
                )
                for goal in goals:
                    await cur.execute(
                        f"""
                        INSERT INTO {self.short_term_goals_table}
                        (id, user_id, session_id, goal, metadata, pinned_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            UUID(goal.id),
                            user_id,
                            session_id,
                            goal.goal,
                            json.dumps(goal.metadata),
                            goal.pinned_at,
                        ),
                    )
            if hasattr(conn, "commit"):
                await conn.commit()

    async def get_pinned_goals(self, user_id: str, session_id: str) -> list[PinnedGoal]:
        async with self._connection_context() as conn, conn.cursor() as cur:
            await cur.execute(
                f"""
                    SELECT id, goal, metadata, pinned_at
                    FROM {self.short_term_goals_table}
                    WHERE user_id = %s AND session_id = %s
                    ORDER BY pinned_at ASC
                    """,
                (user_id, session_id),
            )
            rows = await cur.fetchall()
        return [
            PinnedGoal(
                id=str(row[0]),
                goal=row[1],
                metadata=row[2] if isinstance(row[2], dict) else json.loads(row[2] or "{}"),
                pinned_at=row[3],
            )
            for row in rows
        ]

    async def prune_session(
        self, user_id: str, session_id: str, max_tokens: int, token_counter: TokenCounter
    ) -> WorkingContext:
        messages = await self.get_messages(user_id, session_id)
        goals = await self.get_pinned_goals(user_id, session_id)

        pinned_cost = sum(
            token_counter.count(f"Pinned instruction: {goal.goal}\n") for goal in goals
        )
        remaining = max_tokens - pinned_cost

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

        return WorkingContext(
            messages=retained,
            pinned_goals=goals,
            total_tokens=pinned_cost + message_cost,
            truncated=truncated,
        )

    async def clear_session(self, user_id: str, session_id: str) -> None:
        async with self._connection_context() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"DELETE FROM {self.short_term_messages_table} WHERE user_id = %s AND session_id = %s",
                    (user_id, session_id),
                )
                await cur.execute(
                    f"DELETE FROM {self.short_term_goals_table} WHERE user_id = %s AND session_id = %s",
                    (user_id, session_id),
                )
                await cur.execute(
                    f"DELETE FROM {self.short_term_sessions_table} WHERE user_id = %s AND session_id = %s",
                    (user_id, session_id),
                )
            if hasattr(conn, "commit"):
                await conn.commit()

    async def is_session_expired(
        self, user_id: str, session_id: str, max_age_seconds: float
    ) -> bool:
        async with self._connection_context() as conn, conn.cursor() as cur:
            await cur.execute(
                f"SELECT last_accessed_at FROM {self.short_term_sessions_table} WHERE user_id = %s AND session_id = %s",
                (user_id, session_id),
            )
            row = await cur.fetchone()
        if not row:
            return True
        last_accessed: datetime = row[0]
        now = datetime.now(UTC)
        return (now - last_accessed).total_seconds() >= max_age_seconds

    # --- Long-Term Operations ---

    async def save_record(self, record: LongTermMemoryRecord) -> LongTermMemoryRecord:
        vector_val = _format_vector(record.embedding)
        sup_uuids = [UUID(uid) for uid in record.supersedes_ids]
        async with self._connection_context() as conn:
            async with conn.cursor() as cur:
                # Mark superseded records in DB
                if sup_uuids:
                    await cur.execute(
                        f"""
                        UPDATE {self.long_term_table}
                        SET is_superseded = true, updated_at = now()
                        WHERE user_id = %s AND id = ANY(%s)
                        """,
                        (record.user_id, sup_uuids),
                    )
                await cur.execute(
                    f"""
                    INSERT INTO {self.long_term_table}
                    (id, user_id, content, category, confidence, supersedes_ids, is_superseded, metadata, created_at, updated_at, expires_at, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        category = EXCLUDED.category,
                        confidence = EXCLUDED.confidence,
                        supersedes_ids = EXCLUDED.supersedes_ids,
                        is_superseded = EXCLUDED.is_superseded,
                        metadata = EXCLUDED.metadata,
                        updated_at = EXCLUDED.updated_at,
                        expires_at = EXCLUDED.expires_at,
                        embedding = EXCLUDED.embedding
                    """,
                    (
                        UUID(record.id),
                        record.user_id,
                        record.content,
                        str(record.category),
                        record.confidence,
                        sup_uuids,
                        record.is_superseded,
                        json.dumps(record.metadata),
                        record.created_at,
                        record.updated_at,
                        record.expires_at,
                        vector_val,
                    ),
                )
            if hasattr(conn, "commit"):
                await conn.commit()
        return record

    async def get_record(self, user_id: str, record_id: str) -> LongTermMemoryRecord | None:
        async with self._connection_context() as conn, conn.cursor() as cur:
            await cur.execute(
                f"""
                SELECT id, user_id, content, category, confidence, supersedes_ids, is_superseded,
                       metadata, created_at, updated_at, expires_at, embedding
                FROM {self.long_term_table}
                WHERE user_id = %s AND id = %s
                """,
                (user_id, UUID(record_id)),
            )
            row = await cur.fetchone()
        if not row:
            return None
        return self._row_to_record(row)

    async def list_records(
        self,
        user_id: str,
        category: MemoryCategory | str | None = None,
        include_superseded: bool = False,
        include_expired: bool = False,
        limit: int = 100,
    ) -> list[LongTermMemoryRecord]:
        query = f"""
        SELECT id, user_id, content, category, confidence, supersedes_ids, is_superseded,
               metadata, created_at, updated_at, expires_at, embedding
        FROM {self.long_term_table}
        WHERE user_id = %s
          AND (%s OR NOT is_superseded)
          AND (%s OR expires_at IS NULL OR expires_at > now())
          AND (%s::text IS NULL OR category = %s)
        ORDER BY created_at DESC
        LIMIT %s
        """
        async with self._connection_context() as conn, conn.cursor() as cur:
            cat_str = str(category) if category is not None else None
            await cur.execute(
                query, (user_id, include_superseded, include_expired, cat_str, cat_str, limit)
            )
            rows = await cur.fetchall()
        return [self._row_to_record(row) for row in rows]

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
        candidate_limit = min(max(top_k * 10, 100), 5000)
        vector_val = _format_vector(query_embedding)
        cat_str = str(category) if category is not None else None

        query = f"""
        WITH candidates AS (
            SELECT m.*,
                1 - (m.embedding <=> %s::vector) AS similarity
            FROM {self.long_term_table} m
            WHERE m.user_id = %s
              AND (m.expires_at IS NULL OR m.expires_at > now())
              AND (%s::text IS NULL OR m.category = %s)
              AND m.embedding IS NOT NULL
            ORDER BY m.embedding <=> %s::vector
            LIMIT %s
        ), scored AS (
            SELECT candidates.*,
                (
                    candidates.is_superseded OR
                    EXISTS (
                        SELECT 1
                        FROM {self.long_term_table} newer
                        WHERE newer.user_id = candidates.user_id
                          AND candidates.id = ANY(newer.supersedes_ids)
                          AND (newer.expires_at IS NULL OR newer.expires_at > now())
                    )
                ) AS eff_superseded
            FROM candidates
        )
        SELECT id, user_id, content, category, confidence, supersedes_ids, eff_superseded,
               metadata, created_at, updated_at, expires_at, embedding,
               similarity,
               exp(-ln(2) * GREATEST(
                   0.0, extract(epoch FROM (now() - created_at)) / 86400.0
               ) / %s) AS recency_score,
               (0.65 * similarity)
               + (0.35 * exp(-ln(2) * GREATEST(
                   0.0, extract(epoch FROM (now() - created_at)) / 86400.0
               ) / %s))
               - CASE WHEN eff_superseded THEN 2.0 ELSE 0.0 END AS rank_score
        FROM scored
        WHERE (%s OR NOT eff_superseded)
          AND similarity >= %s
        ORDER BY rank_score DESC, created_at DESC
        LIMIT %s
        """
        async with self._connection_context() as conn, conn.cursor() as cur:
            await cur.execute(
                query,
                (
                    vector_val,
                    user_id,
                    cat_str,
                    cat_str,
                    vector_val,
                    candidate_limit,
                    half_life_days,
                    half_life_days,
                    include_superseded,
                    similarity_threshold,
                    top_k,
                ),
            )
            rows = await cur.fetchall()

        results: list[MemorySearchResult] = []
        for row in rows:
            record = self._row_to_record(row[:12])
            similarity = float(row[12])
            recency = float(row[13])
            rank = float(row[14])
            results.append(
                MemorySearchResult(
                    record=record,
                    similarity=round(similarity, 4),
                    recency_score=round(recency, 4),
                    rank_score=round(rank, 4),
                )
            )
        return results

    async def supersede_records(self, user_id: str, superseded_ids: list[str]) -> int:
        if not superseded_ids:
            return 0
        sup_uuids = [UUID(uid) for uid in superseded_ids]
        async with self._connection_context() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    UPDATE {self.long_term_table}
                    SET is_superseded = true, updated_at = now()
                    WHERE user_id = %s AND id = ANY(%s) AND is_superseded = false
                    """,
                    (user_id, sup_uuids),
                )
                count = cur.rowcount
            if hasattr(conn, "commit"):
                await conn.commit()
        return max(count, 0)

    async def delete_record(self, user_id: str, record_id: str) -> bool:
        async with self._connection_context() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"DELETE FROM {self.long_term_table} WHERE user_id = %s AND id = %s",
                    (user_id, UUID(record_id)),
                )
                deleted = cur.rowcount > 0
            if hasattr(conn, "commit"):
                await conn.commit()
        return deleted

    async def purge_expired(self, user_id: str | None = None) -> int:
        async with self._connection_context() as conn:
            async with conn.cursor() as cur:
                if user_id:
                    await cur.execute(
                        f"DELETE FROM {self.long_term_table} WHERE user_id = %s AND expires_at <= now()",
                        (user_id,),
                    )
                else:
                    await cur.execute(
                        f"DELETE FROM {self.long_term_table} WHERE expires_at <= now()"
                    )
                count = cur.rowcount
            if hasattr(conn, "commit"):
                await conn.commit()
        return max(count, 0)

    # --- Episodic Operations ---

    async def record_episode(self, episode: ActionEpisode) -> ActionEpisode:
        async with self._connection_context() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    INSERT INTO {self.episodic_table}
                    (id, user_id, session_id, action_type, parameters, parameter_hash, result_summary, status, occurred_at, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        UUID(episode.id),
                        episode.user_id,
                        episode.session_id,
                        episode.action_type,
                        json.dumps(episode.parameters),
                        episode.parameter_hash,
                        episode.result_summary,
                        str(episode.status),
                        episode.occurred_at,
                        json.dumps(episode.metadata),
                    ),
                )
            if hasattr(conn, "commit"):
                await conn.commit()
        return episode

    async def get_episodes(
        self,
        user_id: str,
        action_type: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[ActionEpisode]:
        query = f"""
        SELECT id, user_id, session_id, action_type, parameters, parameter_hash,
               result_summary, status, occurred_at, metadata
        FROM {self.episodic_table}
        WHERE user_id = %s
          AND (%s::text IS NULL OR action_type = %s)
          AND (%s::text IS NULL OR session_id = %s)
        ORDER BY occurred_at DESC
        LIMIT %s
        """
        async with self._connection_context() as conn, conn.cursor() as cur:
            await cur.execute(
                query, (user_id, action_type, action_type, session_id, session_id, limit)
            )
            rows = await cur.fetchall()
        return [self._row_to_episode(row) for row in rows]

    async def find_matching_episode(
        self,
        user_id: str,
        action_type: str,
        parameter_hash: str,
        session_id: str | None = None,
    ) -> ActionEpisode | None:
        query = f"""
        SELECT id, user_id, session_id, action_type, parameters, parameter_hash,
               result_summary, status, occurred_at, metadata
        FROM {self.episodic_table}
        WHERE user_id = %s
          AND action_type = %s
          AND parameter_hash = %s
          AND (%s::text IS NULL OR session_id = %s)
          AND (expires_at IS NULL OR expires_at > now())
        ORDER BY occurred_at DESC
        LIMIT 1
        """
        async with self._connection_context() as conn, conn.cursor() as cur:
            await cur.execute(query, (user_id, action_type, parameter_hash, session_id, session_id))
            row = await cur.fetchone()
        if not row:
            return None
        return self._row_to_episode(row)

    async def delete_episodes(self, user_id: str) -> int:
        async with self._connection_context() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"DELETE FROM {self.episodic_table} WHERE user_id = %s", (user_id,)
                )
                count = cur.rowcount
            if hasattr(conn, "commit"):
                await conn.commit()
        return max(count, 0)

    async def purge_episodes(self, older_than: datetime, user_id: str | None = None) -> int:
        async with self._connection_context() as conn:
            async with conn.cursor() as cur:
                if user_id:
                    await cur.execute(
                        f"DELETE FROM {self.episodic_table} WHERE user_id = %s AND occurred_at < %s",
                        (user_id, older_than),
                    )
                else:
                    await cur.execute(
                        f"DELETE FROM {self.episodic_table} WHERE occurred_at < %s", (older_than,)
                    )
                count = cur.rowcount
            if hasattr(conn, "commit"):
                await conn.commit()
        return max(count, 0)

    @staticmethod
    def _row_to_record(row: tuple[Any, ...]) -> LongTermMemoryRecord:
        raw_emb = row[11]
        emb: list[float] | None = None
        if raw_emb is not None:
            if hasattr(raw_emb, "to_list"):
                emb = [float(x) for x in raw_emb.to_list()]
            elif isinstance(raw_emb, (list, tuple)):
                emb = [float(x) for x in raw_emb]
            elif isinstance(raw_emb, str):
                emb = [float(x) for x in raw_emb.strip("[]").split(",") if x.strip()]

        sup_ids: list[str] = []
        if row[5]:
            sup_ids = [str(x) for x in row[5]]

        return LongTermMemoryRecord(
            id=str(row[0]),
            user_id=row[1],
            content=row[2],
            category=row[3],
            confidence=float(row[4]),
            supersedes_ids=sup_ids,
            is_superseded=bool(row[6]),
            metadata=row[7] if isinstance(row[7], dict) else json.loads(row[7] or "{}"),
            created_at=row[8],
            updated_at=row[9],
            expires_at=row[10],
            embedding=emb,
        )

    @staticmethod
    def _row_to_episode(row: tuple[Any, ...]) -> ActionEpisode:
        return ActionEpisode(
            id=str(row[0]),
            user_id=row[1],
            session_id=row[2],
            action_type=row[3],
            parameters=row[4] if isinstance(row[4], dict) else json.loads(row[4] or "{}"),
            parameter_hash=row[5],
            result_summary=row[6],
            status=row[7],
            occurred_at=row[8],
            metadata=row[9] if isinstance(row[9], dict) else json.loads(row[9] or "{}"),
        )
