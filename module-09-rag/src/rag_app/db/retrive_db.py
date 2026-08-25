from time import perf_counter
from uuid import UUID

import psycopg
from pgvector.psycopg import register_vector_async

from rag_app.core import get_settings
from rag_app.tracker.db_operation_tracker import DBOperationTracker
from rag_app.models import RetriveResult
from rag_app.models.tracker.db_query_tracker import DBQueryTracker
from rag_app.core.config import DATABASE_CONNECTION_CONVERSATION_URL


class RetriveDBManager:
    def __init__(self):
        self.settings = get_settings()
        self.db_tracker = DBOperationTracker()

    async def retrive_chunks(
        self,
        tenant_id: UUID,
        query_embedding: list[float],
        top_k: int | None = None,
        document_type: str | None = None,
    ):
        if document_type is None:
            db_query = """
                SELECT
                    chunk_id,
                    chunk_text,
                    1 - (embedding <=> %s) AS similarity_score
                FROM document_chunks
                WHERE tenant_id = %s
                ORDER BY embedding <=> %s
                LIMIT %s
            """

            query_params: tuple[object, ...] = (
                query_embedding,
                tenant_id,
                query_embedding,
                top_k,
            )

        else:
            db_query = """
                SELECT
                    chunk_id,
                    chunk_text,
                    1 - (embedding <=> %s) AS similarity_score
                FROM document_chunks
                WHERE tenant_id = %s
                  AND document_type = %s
                ORDER BY embedding <=> %s
                LIMIT %s
            """

            query_params = (
                query_embedding,
                tenant_id,
                document_type,
                query_embedding,
                top_k,
            )

        query_start = perf_counter()

        async with await psycopg.AsyncConnection.connect(
            DATABASE_CONNECTION_CONVERSATION_URL
        ) as conn:
            await register_vector_async(conn)

            async with conn.cursor() as cur:
                await cur.execute(
                    db_query,
                    query_params,
                )

                rows = await cur.fetchall()

        query_latency_ms = (perf_counter() - query_start) * 1000

        results = [
            RetriveResult(
                chunk_id=row[0],
                chunk_text=row[1],
                similarity_score=float(row[2]),
            )
            for row in rows
        ]

        # Track query execution
        db_query_tracker = DBQueryTracker(
            tenant_id=str(tenant_id),
            top_k=top_k or self.settings.default_top_k,
            results_count=len(results),
            query_latency_ms=query_latency_ms,
            chunk_ids=[id[0] for id in rows],
        )
        self.db_tracker.track_query(db_query_tracker)

        return results

    async def retrive_keyword_chunks(
        self,
        tenant_id: UUID,
        query: str,
        top_k: int | None = None,
        document_type: str | None = None,
    ):
        if document_type is None:
            db_query = """
                    SELECT
                        chunk_id,
                        chunk_text,
                        ts_rank_cd(
                            search_vector,
                            websearch_to_tsquery('english', %s)
                        ) AS keyword_score
                    FROM document_chunks
                    WHERE tenant_id = %s
                    AND search_vector @@ websearch_to_tsquery('english', %s)
                    ORDER BY keyword_score DESC
                    LIMIT %s
                """

            query_params: tuple[object, ...] = (
                query,
                tenant_id,
                query,
                top_k,
            )

        else:
            db_query = """
                    SELECT
                        chunk_id,
                        chunk_text,
                        ts_rank_cd(
                            search_vector,
                            websearch_to_tsquery('english', %s)
                        ) AS keyword_score
                    FROM document_chunks
                    WHERE tenant_id = %s
                    AND document_type = %s
                    AND search_vector @@ websearch_to_tsquery('english', %s)
                    ORDER BY keyword_score DESC
                    LIMIT %s
                """

            query_params = (
                query,
                tenant_id,
                document_type,
                query,
                top_k,
            )

        query_start = perf_counter()

        async with await psycopg.AsyncConnection.connect(
            DATABASE_CONNECTION_CONVERSATION_URL
        ) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    db_query,
                    query_params,
                )

                rows = await cur.fetchall()

        query_latency_ms = (perf_counter() - query_start) * 1000

        results = [
            RetriveResult(
                chunk_id=row[0],
                chunk_text=row[1],
                similarity_score=float(row[2]),
            )
            for row in rows
        ]

        db_query_tracker = DBQueryTracker(
            tenant_id=str(tenant_id),
            top_k=top_k or self.settings.default_top_k,
            results_count=len(results),
            query_latency_ms=query_latency_ms,
            chunk_ids=[row[0] for row in rows],
        )

        self.db_tracker.track_query(db_query_tracker)

        return results
