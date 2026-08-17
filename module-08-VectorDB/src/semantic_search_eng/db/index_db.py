import psycopg
from pgvector.psycopg import register_vector_async
from uuid import UUID
from semantic_search_eng.models.chunk import Chunk

from semantic_search_eng.config import get_settings

from psycopg.types.json import Jsonb


class IndexDBManager:
    BATCH_SIZE = 1000

    def __init__(self):
        self.settings = get_settings()

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
                for i in range(0, len(rows), self.BATCH_SIZE):
                    batch = rows[i : i + self.BATCH_SIZE]
                    await cur.executemany(query, batch)
                    # add index db tracker here

            await conn.commit()
