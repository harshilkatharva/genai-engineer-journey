from uuid import uuid4

import pytest
import psycopg

from rag_app.db.retrive_db import RetriveDBManager
from rag_app.core.config import DATABASE_CONNECTION_CONVERSATION_URL


@pytest.mark.asyncio
async def test_keyword_search_code_name():
    tenant_id = uuid4()

    async with await psycopg.AsyncConnection.connect(DATABASE_CONNECTION_CONVERSATION_URL) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO document_chunks (
                    tenant_id,
                    document_id,
                    chunk_id,
                    chunk_text,
                    embedding,
                    document_type
                )
                VALUES
                    (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s
                    )
                """,
                (
                    tenant_id,
                    uuid4(),
                    "test-code-1",
                    "Employee EMP-1024 is assigned to the PAY-202 payroll process.",
                    [0.0] * 384,
                    "hr",
                ),
            )

        await conn.commit()

    manager = RetriveDBManager()

    results = await manager.retrive_keyword_chunks(
        tenant_id=tenant_id,
        query="EMP-1024",
        top_k=5,
    )

    assert any(result.chunk_id == "test-code-1" for result in results)
