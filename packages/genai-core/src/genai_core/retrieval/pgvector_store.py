from __future__ import annotations

import contextlib
import inspect
import json
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, NoReturn

from ..exceptions import (
    IncompatibleEmbeddingError,
    MissingExtraDependencyError,
    VectorStoreError,
)
from ..models.retrieval_model import DocumentChunk, RetrievalResult

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


def check_postgres_dependencies() -> None:
    """Verify that optional PostgreSQL and pgvector libraries are installed."""
    try:
        import pgvector  # type: ignore[import-not-found]  # noqa: F401
        import psycopg  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as err:
        raise MissingExtraDependencyError(
            extra="postgres",
            package="psycopg[binary] and pgvector",
            feature="PGVectorStoreAdapter",
        ) from err


def _to_vector_param(vec: list[float]) -> Any:
    """Wrap float list into pgvector Vector with sequence equality for psycopg and tests."""
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


class PGVectorStoreAdapter:
    """PostgreSQL + pgvector vector store adapter conforming to VectorStoreAdapter protocol."""

    def __init__(
        self,
        *,
        dimensions: int,
        connection: Any = None,
        connection_pool: Any = None,
        connection_factory: Callable[[], Any] | None = None,
        table_name: str = "document_chunks",
        embedding_column: str = "embedding",
        id_column: str = "chunk_id",
        text_column: str = "chunk_text",
        metadata_column: str = "metadata",
        tenant_id_column: str | None = None,
        conflict_target: str | None = None,
        search_vector_column: str | None = None,
        language: str = "english",
        check_dependencies: bool = False,
    ) -> None:
        if dimensions <= 0:
            raise ValueError(f"dimensions must be greater than 0, got {dimensions}")
        self.dimensions = dimensions

        if check_dependencies:
            check_postgres_dependencies()

        if connection is None and connection_pool is None and connection_factory is None:
            if not check_dependencies:
                # If neither connection nor dependencies were checked, check them now
                check_postgres_dependencies()
            raise ValueError(
                "PGVectorStoreAdapter requires an injected 'connection', 'connection_pool', "
                "or 'connection_factory'."
            )

        self._connection = connection
        self._connection_pool = connection_pool
        self._connection_factory = connection_factory

        self.table_name = _validate_identifier(table_name, "table_name")
        self.embedding_column = _validate_identifier(embedding_column, "embedding_column")
        self.id_column = _validate_identifier(id_column, "id_column")
        self.text_column = _validate_identifier(text_column, "text_column")
        self.metadata_column = _validate_identifier(metadata_column, "metadata_column")

        self.tenant_id_column = (
            _validate_identifier(tenant_id_column, "tenant_id_column")
            if tenant_id_column is not None
            else None
        )
        self.conflict_target = (
            _validate_identifier(conflict_target, "conflict_target")
            if conflict_target is not None
            else None
        )
        self.search_vector_column = (
            _validate_identifier(search_vector_column, "search_vector_column")
            if search_vector_column is not None
            else None
        )
        self.language = _validate_identifier(language, "language")

    @asynccontextmanager
    async def _get_connection(self) -> AsyncIterator[Any]:
        """Yield an active async database connection from the pool, factory, or instance."""
        if self._connection_pool is not None:
            async with self._connection_pool.connection() as conn:
                yield conn
        elif self._connection_factory is not None:
            conn = self._connection_factory()
            if hasattr(conn, "__aenter__"):
                async with conn as c:
                    yield c
            elif inspect.iscoroutine(conn):
                c = await conn
                yield c
            else:
                yield conn
        elif self._connection is not None:
            yield self._connection
        else:
            raise VectorStoreError(
                "No database connection available.",
                backend="pgvector",
            )

    async def _register_vector_if_needed(self, conn: Any) -> None:
        """Register pgvector type handlers on the connection if pgvector is installed."""
        with contextlib.suppress(Exception):
            from pgvector.psycopg import register_vector_async  # type: ignore[import-not-found]

            await register_vector_async(conn)

    def _handle_db_error(self, err: Exception, operation: str) -> NoReturn:
        """Normalize database exceptions into typed VectorStoreError with retryable hints."""
        if isinstance(err, (ValueError, IncompatibleEmbeddingError, MissingExtraDependencyError)):
            raise err

        err_msg = str(err).lower()
        is_transient = any(
            term in err_msg
            for term in (
                "connection reset",
                "connection refused",
                "connection closed",
                "terminating connection",
                "the connection is closed",
                "broken pipe",
                "timeout",
                "temporarily unavailable",
                "deadlock detected",
            )
        ) or isinstance(err, (ConnectionError, TimeoutError))

        raise VectorStoreError(
            f"PostgreSQL/pgvector {operation} failed: {err}",
            backend="pgvector",
            retryable=is_transient,
        ) from err

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievalResult]:
        """Perform dense semantic vector search with cosine similarity and metadata filtering."""
        if len(query_vector) != self.dimensions:
            raise IncompatibleEmbeddingError(
                requested=len(query_vector),
                supported=self.dimensions,
                message=(
                    f"Query vector dimension ({len(query_vector)}) does not match "
                    f"PGVector store dimension ({self.dimensions})."
                ),
            )

        if top_k <= 0:
            return []

        vec = _to_vector_param(query_vector)
        where_clauses: list[str] = []
        params: list[Any] = [vec]

        active_filters = dict(filters) if filters else {}

        if self.tenant_id_column and "tenant_id" in active_filters:
            where_clauses.append(f"{self.tenant_id_column} = %s")
            params.append(active_filters.pop("tenant_id"))

        if active_filters:
            where_clauses.append(f"{self.metadata_column} @> %s::jsonb")
            params.append(json.dumps(active_filters))

        if score_threshold is not None:
            where_clauses.append(f"1 - ({self.embedding_column} <=> %s) >= %s")
            params.extend([vec, score_threshold])

        params.extend([vec, top_k])

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        sql = f"""
            SELECT
                {self.id_column},
                {self.text_column},
                {self.metadata_column},
                1 - ({self.embedding_column} <=> %s) AS similarity_score
            FROM {self.table_name}
            {where_sql}
            ORDER BY {self.embedding_column} <=> %s ASC
            LIMIT %s
        """

        try:
            async with self._get_connection() as conn:
                await self._register_vector_if_needed(conn)
                async with conn.cursor() as cur:
                    await cur.execute(sql, tuple(params))
                    rows = await cur.fetchall()
        except Exception as err:  # noqa: BLE001 - database adapter boundary converts all failures
            self._handle_db_error(err, "search")

        results: list[RetrievalResult] = []
        for rank, row in enumerate(rows, start=1):
            chunk_id = str(row[0])
            content = str(row[1])
            raw_meta = row[2]
            if isinstance(raw_meta, str):
                try:
                    metadata = json.loads(raw_meta)
                except (json.JSONDecodeError, TypeError, ValueError):
                    metadata = {}
            elif isinstance(raw_meta, dict):
                metadata = raw_meta
            else:
                metadata = {}
            score = float(row[3])
            results.append(
                RetrievalResult(
                    id=chunk_id,
                    content=content,
                    metadata=metadata,
                    score=score,
                    rank=rank,
                )
            )

        return results

    async def search_keyword(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievalResult]:
        """Perform full-text keyword search using PostgreSQL ts_rank_cd and websearch_to_tsquery."""
        if top_k <= 0:
            return []

        if not query.strip():
            return []

        where_clauses: list[str] = []
        select_params: list[Any] = []
        where_params: list[Any] = []

        active_filters = dict(filters) if filters else {}

        if self.tenant_id_column and "tenant_id" in active_filters:
            where_clauses.append(f"{self.tenant_id_column} = %s")
            where_params.append(active_filters.pop("tenant_id"))

        if active_filters:
            where_clauses.append(f"{self.metadata_column} @> %s::jsonb")
            where_params.append(json.dumps(active_filters))

        if self.search_vector_column:
            select_rank_expr = (
                f"ts_rank_cd({self.search_vector_column}, websearch_to_tsquery(%s, %s))"
            )
            select_params.extend([self.language, query])
            where_clauses.append(f"{self.search_vector_column} @@ websearch_to_tsquery(%s, %s)")
            where_params.extend([self.language, query])
        else:
            select_rank_expr = (
                f"ts_rank_cd(to_tsvector(%s, {self.text_column}), websearch_to_tsquery(%s, %s))"
            )
            select_params.extend([self.language, self.language, query])
            where_clauses.append(
                f"to_tsvector(%s, {self.text_column}) @@ websearch_to_tsquery(%s, %s)"
            )
            where_params.extend([self.language, self.language, query])

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        outer_where: list[str] = []
        outer_params: list[Any] = []
        if score_threshold is not None:
            outer_where.append("keyword_score >= %s")
            outer_params.append(score_threshold)

        outer_where_sql = f"WHERE {' AND '.join(outer_where)}" if outer_where else ""

        sql = f"""
            WITH matched_chunks AS (
                SELECT
                    {self.id_column},
                    {self.text_column},
                    {self.metadata_column},
                    {select_rank_expr} AS keyword_score
                FROM {self.table_name}
                {where_sql}
            )
            SELECT
                {self.id_column},
                {self.text_column},
                {self.metadata_column},
                keyword_score
            FROM matched_chunks
            {outer_where_sql}
            ORDER BY keyword_score DESC
            LIMIT %s
        """

        all_params = tuple(select_params + where_params + outer_params + [top_k])

        try:
            async with self._get_connection() as conn, conn.cursor() as cur:
                await cur.execute(sql, all_params)
                rows = await cur.fetchall()
        except Exception as err:  # noqa: BLE001 - database adapter boundary converts all failures
            self._handle_db_error(err, "keyword search")

        results: list[RetrievalResult] = []
        for rank, row in enumerate(rows, start=1):
            chunk_id = str(row[0])
            content = str(row[1])
            raw_meta = row[2]
            if isinstance(raw_meta, str):
                try:
                    metadata = json.loads(raw_meta)
                except (json.JSONDecodeError, TypeError, ValueError):
                    metadata = {}
            elif isinstance(raw_meta, dict):
                metadata = raw_meta
            else:
                metadata = {}
            score = float(row[3])
            results.append(
                RetrievalResult(
                    id=chunk_id,
                    content=content,
                    metadata=metadata,
                    score=round(score, 6),
                    rank=rank,
                )
            )

        return results

    async def upsert(self, chunks: list[DocumentChunk]) -> int:
        """Insert or update chunks and their embeddings in PostgreSQL/pgvector."""
        if not chunks:
            return 0

        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk '{chunk.id}' is missing embedding vector.")
            if len(chunk.embedding) != self.dimensions:
                raise IncompatibleEmbeddingError(
                    requested=len(chunk.embedding),
                    supported=self.dimensions,
                    message=(
                        f"Chunk '{chunk.id}' embedding dimension ({len(chunk.embedding)}) "
                        f"does not match PGVector store dimension ({self.dimensions})."
                    ),
                )

        conflict_col = self.conflict_target or self.id_column
        sql = f"""
            INSERT INTO {self.table_name} (
                {self.id_column},
                {self.text_column},
                {self.embedding_column},
                {self.metadata_column}
            )
            VALUES (%s, %s, %s, %s::jsonb)
            ON CONFLICT ({conflict_col})
            DO UPDATE SET
                {self.text_column} = EXCLUDED.{self.text_column},
                {self.embedding_column} = EXCLUDED.{self.embedding_column},
                {self.metadata_column} = EXCLUDED.{self.metadata_column}
        """

        rows: list[tuple[Any, ...]] = []
        for chunk in chunks:
            assert chunk.embedding is not None
            rows.append(
                (
                    chunk.id,
                    chunk.content,
                    _to_vector_param(chunk.embedding),
                    json.dumps(chunk.metadata),
                )
            )

        try:
            async with self._get_connection() as conn:
                await self._register_vector_if_needed(conn)
                async with conn.cursor() as cur:
                    await cur.executemany(sql, rows)
                if hasattr(conn, "commit"):
                    res = conn.commit()
                    if inspect.iscoroutine(res):
                        await res
            return len(chunks)
        except Exception as err:  # noqa: BLE001 - database adapter boundary converts all failures
            self._handle_db_error(err, "upsert")

    async def delete(self, chunk_ids: list[str]) -> int:
        """Delete chunks by their IDs from PostgreSQL/pgvector."""
        if not chunk_ids:
            return 0

        sql = f"DELETE FROM {self.table_name} WHERE {self.id_column} = ANY(%s)"

        try:
            async with self._get_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(sql, (list(chunk_ids),))
                    rowcount = cur.rowcount if cur.rowcount >= 0 else len(chunk_ids)
                if hasattr(conn, "commit"):
                    res = conn.commit()
                    if inspect.iscoroutine(res):
                        await res
            return rowcount
        except Exception as err:  # noqa: BLE001 - database adapter boundary converts all failures
            self._handle_db_error(err, "delete")


__all__ = [
    "PGVectorStoreAdapter",
    "check_postgres_dependencies",
]
