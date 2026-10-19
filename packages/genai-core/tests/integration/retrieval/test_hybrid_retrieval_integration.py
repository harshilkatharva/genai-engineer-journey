from __future__ import annotations

import contextlib
import os
from collections.abc import AsyncGenerator

import pytest

from genai_core.embedding import LocalEmbeddingProvider
from genai_core.models import DocumentChunk, RetrievalQuery
from genai_core.retrieval import CrossEncoderReranker
from genai_core.retrieval.pgvector_store import PGVectorStoreAdapter
from genai_core.services import EmbeddingService, RetrievalService

POSTGRES_URL = (
    os.getenv("DATABASE_CONNECTION_CONVERSATION_URL")
    or os.getenv("POSTGRES_URL")
    or "postgresql://postgres:postgres@127.0.0.1:5432/ai_search"
)

TEST_HYBRID_TABLE = "test_genai_core_hybrid_chunks"


@pytest.fixture
async def pg_hybrid_store() -> AsyncGenerator[PGVectorStoreAdapter, None]:
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
                CREATE TABLE IF NOT EXISTS {TEST_HYBRID_TABLE} (
                    chunk_id TEXT PRIMARY KEY,
                    chunk_text TEXT NOT NULL,
                    embedding vector(384) NOT NULL,
                    metadata JSONB DEFAULT '{{}}'::jsonb
                );
                """
            )
            await cur.execute(f"TRUNCATE {TEST_HYBRID_TABLE};")

        store = PGVectorStoreAdapter(
            dimensions=384,
            connection=conn,
            table_name=TEST_HYBRID_TABLE,
        )
        yield store

    finally:
        with contextlib.suppress(Exception):
            async with conn.cursor() as cur:
                await cur.execute(f"DROP TABLE IF EXISTS {TEST_HYBRID_TABLE};")
            await conn.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_hybrid_search_with_reranking(
    pg_hybrid_store: PGVectorStoreAdapter,
) -> None:
    embed_provider = LocalEmbeddingProvider(model="all-MiniLM-L6-v2")
    embed_service = EmbeddingService(provider=embed_provider)
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L6-v2")

    retrieval_service = RetrievalService(
        store=pg_hybrid_store,
        embedding_service=embed_service,
        reranker=reranker,
        enable_reranking=True,
    )

    documents = [
        "Cloud computing infrastructure provides scalable, on-demand compute and storage solutions.",
        "Internal project ticket PROJ-999 is assigned to the hybrid enterprise cloud migration team.",
        "Classic Italian pizza napoletana requires high-gluten flour, San Marzano tomatoes, and mozzarella.",
    ]
    vectors = await embed_service.embed_documents(documents)

    chunks = [
        DocumentChunk(id="cloud_doc", content=documents[0], embedding=vectors[0]),
        DocumentChunk(id="proj_doc", content=documents[1], embedding=vectors[1]),
        DocumentChunk(id="food_doc", content=documents[2], embedding=vectors[2]),
    ]
    await pg_hybrid_store.upsert(chunks)

    # Perform hybrid search with query containing exact code and semantic meaning
    query = RetrievalQuery(
        query="PROJ-999 enterprise cloud migration",
        search_mode="hybrid",
        top_k=2,
        rerank=True,
        rerank_top_k=2,
    )

    response = await retrieval_service.search(query)

    assert response.search_mode == "hybrid"
    assert response.reranked is True
    assert len(response.results) == 2

    # proj_doc contains the exact code PROJ-999 and cloud migration topic
    assert response.results[0].id == "proj_doc"
    assert response.results[0].rank == 1
    assert "retrieval_score" in response.results[0].metadata
