from time import perf_counter
from uuid import UUID

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg.types.json import Jsonb

from rag_app.core import get_settings
from rag_app.models.chunk.chunk import Chunk
from rag_app.models.tracker.index_batch_tracker import IndexBatchTracker
from rag_app.tracker.db_operation_tracker import DBOperationTracker


class IndexDBManager:
    BATCH_SIZE = 1000

    def __init__(self):
        self.settings = get_settings()
        self.db_tracker = DBOperationTracker()

    async def store_index(self, tenant_id: UUID, chunks: list[Chunk], embeddings: list):
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        rows = [
            (
                tenant_id,
                chunk.document_id,
                chunk.chunk_id,
                chunk.text,
                embedding,
                chunk.document_type,
                Jsonb(chunk.metadata),
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        query = """
            INSERT INTO document_chunks (
                tenant_id,
                document_id,
                chunk_id,
                chunk_text,
                embedding,
                document_type,
                metadata
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s
            )
        """

        async with await psycopg.AsyncConnection.connect(
            self.settings.DATABASE_CONNECTION_CONVERSATION_URL
        ) as conn:
            await register_vector_async(conn)

            async with conn.cursor() as cur:
                for batch_num, i in enumerate(range(0, len(rows), self.BATCH_SIZE), 1):
                    batch = rows[i : i + self.BATCH_SIZE]
                    batch_start = perf_counter()
                    await cur.executemany(query, batch)
                    insertion_latency_ms = (perf_counter() - batch_start) * 1000

                    # Track index batch insertion
                    index_tracker = IndexBatchTracker(
                        tenant_id=str(tenant_id),
                        batch_number=batch_num,
                        batch_size=len(batch),
                        total_chunks=len(rows),
                        insertion_latency_ms=insertion_latency_ms,
                    )
                    self.db_tracker.track_index_batch(index_tracker)

            await conn.commit()
