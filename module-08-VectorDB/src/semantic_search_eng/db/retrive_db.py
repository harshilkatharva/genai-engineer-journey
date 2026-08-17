from semantic_search_eng.models import RetriveResult
import psycopg
from pgvector.psycopg import register_vector_async
from semantic_search_eng.config import get_settings
from uuid import UUID


class RetriveDBManager:
    def __init__(self):
        self.settings = get_settings()

    async def retrive_chunks(
        self, tenant_id: UUID, query_embedding: list[float], top_k: int | None = None
    ):
        db_query = """
        SELECT chunk_text, 1 - (embedding <=> %s) as similarity_score
        FROM document_chunks
        WHERE tenant_id = %s
        ORDER BY embedding <=> %s 
        LIMIT %s
        """

        async with await psycopg.AsyncConnection.connect(
            self.settings.DATABASE_CONNECTION_CONVERSATION_URL
        ) as conn:
            await register_vector_async(conn)

            async with conn.cursor() as cur:
                await cur.execute(
                    db_query,
                    (
                        query_embedding,
                        tenant_id,
                        query_embedding,
                        top_k,
                    ),
                )

                rows = await cur.fetchall()

        results = [
            RetriveResult(
                chunk_text=row[0],
                similarity_score=float(row[1]),
            )
            for row in rows
        ]

        return results
