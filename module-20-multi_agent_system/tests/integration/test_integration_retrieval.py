"""Integration test for RetrievalService with in-memory store and PostgreSQL database connectivity."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
from genai_core.embedding.local_provider import LocalEmbeddingProvider
from genai_core.models import DocumentChunk
from genai_core.retrieval.in_memory_store import InMemoryVectorStore
from genai_core.services.embedding_service import EmbeddingService
from genai_core.services.retrieval_service import RetrievalService

from parallel_research_team.core.config import ResearchTeamConfig
from parallel_research_team.workers.tools import create_retrieval_service

POSTGRES_URL = (
    os.getenv("POSTGRES_CONNECTION_STRING")
    or os.getenv("POSTGRES_URL")
    or "postgresql://postgres:postgres@127.0.0.1:5432/ai_search"
)

FIXTURES_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "research_chunks.json"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_in_memory_retrieval_service_with_real_chunks() -> None:
    """Verify in-memory retrieval service with real handbook chunks and local embeddings."""
    if not FIXTURES_PATH.exists():
        pytest.skip(f"Fixture file not found: {FIXTURES_PATH}")

    content = await asyncio.to_thread(FIXTURES_PATH.read_text, encoding="utf-8")
    raw_chunks: list[dict[str, Any]] = json.loads(content)[:100]

    document_chunks: list[DocumentChunk] = []
    for c in raw_chunks:
        raw_emb = c.get("embedding")
        if isinstance(raw_emb, str):
            emb: list[float] = json.loads(raw_emb)
        else:
            emb = raw_emb or []

        document_chunks.append(
            DocumentChunk(
                id=c["chunk_id"],
                content=c["chunk_text"],
                embedding=emb,
                metadata=c.get("metadata") or {},
                source_id=c.get("metadata", {}).get("document_name", "production_handbook"),
            )
        )

    store = InMemoryVectorStore(dimensions=384, chunks=document_chunks)
    embedding_provider = LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    embedding_service = EmbeddingService(provider=embedding_provider)

    retrieval_service = RetrievalService(
        store=store,
        embedding_service=embedding_service,
    )

    query = "Production GenAI & Agentic AI Engineer Handbook learning roadmap"
    response = await retrieval_service.search(query=query, top_k=3)

    assert response is not None
    assert len(response.results) >= 1

    top_result = response.results[0]
    assert top_result.id != ""
    assert (
        "GENAI" in top_result.content
        or "Handbook" in top_result.content
        or "Engineer" in top_result.content
    )
    assert top_result.score > 0.4
    assert isinstance(top_result.metadata, dict)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_postgres_retrieval_service_connectivity_and_data() -> None:
    """Verify PostgreSQL database connectivity and retrieval against real document_chunks data."""
    try:
        import psycopg
    except ImportError:
        pytest.skip("psycopg is not installed.")

    # Probe live connection to PostgreSQL container
    try:
        conn = await psycopg.AsyncConnection.connect(
            POSTGRES_URL, autocommit=True, connect_timeout=3
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(
            f"PostgreSQL database not reachable at {POSTGRES_URL} (container may be stopped): {exc}"
        )

    try:
        # Check if table exists and has rows
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = 'document_chunks'
                );
                """
            )
            row = await cur.fetchone()
            table_exists = row[0] if row else False

            if not table_exists:
                pytest.skip("Table 'document_chunks' does not exist in PostgreSQL database.")

            await cur.execute("SELECT count(*) FROM document_chunks;")
            count_row = await cur.fetchone()
            total_chunks = count_row[0] if count_row else 0
            if total_chunks == 0:
                pytest.skip("Table 'document_chunks' has 0 rows in PostgreSQL database.")

        # Test PGVectorStoreAdapter via create_retrieval_service factory
        config = ResearchTeamConfig(
            retrieval_mode="postgres",
            postgres_connection_string=POSTGRES_URL,
            embedding_dimensions=384,
        )
        embedding_provider = LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
        embedding_service = EmbeddingService(provider=embedding_provider)

        retrieval_service = create_retrieval_service(
            config=config,
            embedding_service=embedding_service,
        )

        query = "Production GenAI & Agentic AI Engineer Handbook"
        response = await retrieval_service.search(query=query, top_k=3)

        assert response is not None
        assert len(response.results) >= 1
        first = response.results[0]
        assert first.id != ""
        assert len(first.content) > 0
        assert first.score > 0.0

    finally:
        await conn.close()
