from time import perf_counter
from uuid import UUID

import psycopg
from pgvector.psycopg import register_vector_async

from rag_app.config import get_settings
from rag_app.logger.db_operation_tracker import DBOperationTracker
from rag_app.models import RetriveResult
from rag_app.models.db_query_tracker import DBQueryTracker


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
        print(document_type)
        if document_type is None:
            db_query = """
                SELECT
                    chunk_text,
                    1 - (embedding <=> %s) AS similarity_score,
                    chunk_id
                FROM document_chunks
                WHERE tenant_id = %s
                ORDER BY embedding <=> %s
                LIMIT %s
            """

            query_params = (
                query_embedding,
                tenant_id,
                query_embedding,
                top_k,
            )

        else:
            db_query = """
                SELECT
                    chunk_text,
                    1 - (embedding <=> %s) AS similarity_score,
                    chunk_id
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
            self.settings.DATABASE_CONNECTION_CONVERSATION_URL
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
                chunk_text=row[0],
                similarity_score=float(row[1]),
            )
            for row in rows
        ]

        # Track query execution
        db_query_tracker = DBQueryTracker(
            tenant_id=str(tenant_id),
            top_k=top_k or self.settings.default_top_k,
            results_count=len(results),
            query_latency_ms=query_latency_ms,
            chunk_ids=[id[2] for id in rows],
        )
        self.db_tracker.track_query(db_query_tracker)

        return results
