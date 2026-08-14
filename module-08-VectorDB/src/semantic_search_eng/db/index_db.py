import psycopg
from pgvector.psycopg import register_vector
from uuid import UUID
from semantic_search_eng.models.chunk import Chunk

from semantic_search_eng.config import get_settings


class IndexDBManager:
    def __init__(self):
        self.settings = get_settings()

    def store_index(self, tenant_id: UUID, chunks: list[Chunk], embeddings: list):
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
                chunk.metadata,
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
                metadata,
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s
            )
        """

        with psycopg.connect(self.settings.DATABASE_CONNECTION_CONVERSATION_URL) as conn:
            register_vector(conn)

            with conn.cursor() as cur:
                cur.executemany(query, rows)

            conn.commit()
