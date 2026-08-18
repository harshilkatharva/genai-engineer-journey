from __future__ import annotations

from time import perf_counter
from uuid import UUID

import psycopg
from pgvector.psycopg import register_vector_async

from semantic_search_eng.config import get_settings
from semantic_search_eng.logger.db_operation_tracker import DBOperationTracker
from semantic_search_eng.models.db_query_tracker import DBQueryTracker


class UpsertDBManager:
    """
    Updates existing indexed chunks in the vector database.

    A row is identified by:

        tenant_id + document_id + chunk_id

    The operation is update-only.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.db_tracker = DBOperationTracker()

    async def upsert_chunks(
        self,
        tenant_id: UUID,
        updated_chunks: list[str],
        updated_embeddings: list[list[float]],
        document_ids: list,
        chunk_ids: list,
    ) -> list[str]:
        """
        Update multiple existing chunks.

        Returns:
            Number of database rows successfully updated.

        Raises:
            ValueError:
                If chunks and embeddings have different lengths.
        """
        if len(updated_chunks) != len(updated_embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        if not updated_chunks:
            return 0

        query = """
            UPDATE document_chunks
            SET
                chunk_text = %s,
                embedding = %s,
            WHERE tenant_id = %s
              AND document_id = %s
              AND chunk_id = %s
        """

        rows = [
            (
                chunk,
                embedding,
                tenant_id,
                document_id,
                chunk_id,
            )
            for chunk, embedding, document_id, chunk_id in zip(
                updated_chunks,
                updated_embeddings,
                document_ids,
                chunk_ids,
                strict=True,
            )
        ]

        query_start = perf_counter()

        updated_chunk_ids: list[str] = []

        async with await psycopg.AsyncConnection.connect(
            self.settings.DATABASE_CONNECTION_CONVERSATION_URL
        ) as conn:
            await register_vector_async(conn)

            async with conn.cursor() as cur:
                for row, c_id in zip(rows, chunk_ids, strict=True):
                    await cur.execute(query, row)
                    if cur.rowcount > 0:
                        updated_chunk_ids.append(str(c_id))
            await conn.commit()

        query_latency_ms = (perf_counter() - query_start) * 1000

        db_query_tracker = DBQueryTracker(
            tenant_id=str(tenant_id),
            top_k=len(updated_chunks),
            results_count=len(updated_chunk_ids),
            query_latency_ms=query_latency_ms,
            chunk_ids=updated_chunk_ids,
        )

        self.db_tracker.track_query(db_query_tracker)

        return updated_chunk_ids
