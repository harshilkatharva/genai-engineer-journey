from __future__ import annotations

import contextlib
import os
from collections.abc import AsyncGenerator

import pytest

from genai_core.embedding import LocalEmbeddingProvider
from genai_core.models import DocumentChunk
from genai_core.retrieval.pgvector_store import PGVectorStoreAdapter

POSTGRES_URL = (
    os.getenv("DATABASE_CONNECTION_CONVERSATION_URL")
    or os.getenv("POSTGRES_URL")
    or "postgresql://postgres:postgres@127.0.0.1:5432/ai_search"
)

TEST_TABLE = "test_genai_core_retrieval_chunks"


@pytest.fixture
async def pg_adapter() -> AsyncGenerator[PGVectorStoreAdapter, None]:
    try:
        import psycopg  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("psycopg is not installed")

    try:
        conn = await psycopg.AsyncConnection.connect(
            POSTGRES_URL, autocommit=True, connect_timeout=3
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Live PostgreSQL instance not reachable at {POSTGRES_URL}: {exc}")

    try:
        async with conn.cursor() as cur:
            await cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TEST_TABLE} (
                    chunk_id TEXT PRIMARY KEY,
                    chunk_text TEXT NOT NULL,
                    embedding vector(384) NOT NULL,
                    metadata JSONB DEFAULT '{{}}'::jsonb
                );
                """
            )
            await cur.execute(f"TRUNCATE {TEST_TABLE};")

        adapter = PGVectorStoreAdapter(
            dimensions=384,
            connection=conn,
            table_name=TEST_TABLE,
        )
        yield adapter

    finally:
        with contextlib.suppress(Exception):
            async with conn.cursor() as cur:
                await cur.execute(f"DROP TABLE IF EXISTS {TEST_TABLE};")
            await conn.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pgvector_upsert_and_vector_search(pg_adapter: PGVectorStoreAdapter) -> None:
    embedder = LocalEmbeddingProvider(model="all-MiniLM-L6-v2")
    texts = [
        "PostgreSQL is an advanced open source object-relational database system.",
        "Python is a versatile programming language used for machine learning and web apps.",
        "The solar system consists of eight planets orbiting the Sun.",
    ]
    vectors = await embedder.embed_batch(texts)

    chunks = [
        DocumentChunk(
            id="c1",
            content=texts[0],
            embedding=vectors[0],
            metadata={"category": "tech", "topic": "database"},
        ),
        DocumentChunk(
            id="c2",
            content=texts[1],
            embedding=vectors[1],
            metadata={"category": "tech", "topic": "programming"},
        ),
        DocumentChunk(
            id="c3",
            content=texts[2],
            embedding=vectors[2],
            metadata={"category": "science", "topic": "astronomy"},
        ),
    ]

    inserted = await pg_adapter.upsert(chunks)
    assert inserted == 3

    # Query about databases
    query_vec = (await embedder.embed_batch(["SQL relational databases and queries"]))[0]
    results = await pg_adapter.search(query_vec, top_k=2)

    assert len(results) == 2
    assert results[0].id == "c1"
    assert results[0].score > 0.4
    assert results[0].rank == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pgvector_keyword_search(pg_adapter: PGVectorStoreAdapter) -> None:
    embedder = LocalEmbeddingProvider(model="all-MiniLM-L6-v2")
    texts = [
        "Order ORD-9012 has been packed and dispatched for delivery.",
        "Employee EMP-4050 received authorization for payroll update.",
    ]
    vectors = await embedder.embed_batch(texts)

    chunks = [
        DocumentChunk(id="ord1", content=texts[0], embedding=vectors[0]),
        DocumentChunk(id="emp1", content=texts[1], embedding=vectors[1]),
    ]
    await pg_adapter.upsert(chunks)

    results = await pg_adapter.search_keyword("ORD-9012", top_k=2)
    assert len(results) >= 1
    assert results[0].id == "ord1"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pgvector_metadata_filtering(pg_adapter: PGVectorStoreAdapter) -> None:
    embedder = LocalEmbeddingProvider(model="all-MiniLM-L6-v2")
    texts = [
        "Company quarterly earnings and revenue report.",
        "Company employee health insurance benefits policy.",
    ]
    vectors = await embedder.embed_batch(texts)

    chunks = [
        DocumentChunk(
            id="fin1", content=texts[0], embedding=vectors[0], metadata={"department": "finance"}
        ),
        DocumentChunk(
            id="hr1", content=texts[1], embedding=vectors[1], metadata={"department": "hr"}
        ),
    ]
    await pg_adapter.upsert(chunks)

    query_vec = (await embedder.embed_batch(["company policies"]))[0]
    results = await pg_adapter.search(query_vec, top_k=5, filters={"department": "hr"})

    assert len(results) == 1
    assert results[0].id == "hr1"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pgvector_delete(pg_adapter: PGVectorStoreAdapter) -> None:
    embedder = LocalEmbeddingProvider(model="all-MiniLM-L6-v2")
    texts = ["Temporary chunk to be deleted."]
    vectors = await embedder.embed_batch(texts)

    chunk = DocumentChunk(id="temp1", content=texts[0], embedding=vectors[0])
    await pg_adapter.upsert([chunk])

    deleted = await pg_adapter.delete(["temp1"])
    assert deleted == 1

    query_vec = (await embedder.embed_batch(["Temporary chunk"]))[0]
    results = await pg_adapter.search(query_vec, top_k=5)
    assert not any(r.id == "temp1" for r in results)
