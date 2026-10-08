import asyncio
import os
from pathlib import Path

import pytest

from rag_app.core.config import Settings
from rag_app.services.rag import RAGService


@pytest.mark.integration
def test_ingestion_and_storage_round_trip(tmp_path: Path) -> None:
    if os.getenv("INTEGRATION_TEST") != "1" and os.getenv("integrGRATION_TEST") != "1":
        pytest.skip("integration tests require INTEGRATION_TEST=1 or integrGRATION_TEST=1 in env")

    if not os.getenv("DATABASE_CONNECTION_PGVECTOR_URL"):
        pytest.skip("DATABASE_CONNECTION_PGVECTOR_URL must be set for integration storage tests")

    source_dir = tmp_path / "documents"
    source_dir.mkdir()
    (source_dir / "sample.txt").write_text(
        "The project stores semantic memory in a Postgres vector index. "
        "This document is intentionally small so ingestion and persistence can be verified quickly."
        "It discusses retrieval quality, chunking, and storage recovery.",
        encoding="utf-8",
    )

    settings = Settings(
        app_name="integration-rag",
        data_dir=source_dir,
        index_storage_dir=tmp_path / "storage",
        postgres_url=os.environ["DATABASE_CONNECTION_PGVECTOR_URL"],
        vector_table_name="integration_rag_vector_test",
        vector_schema_name="public",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
        reranker_model="cross-encoder/ms-marco-MiniLM-L6-v2",
        reranker_top_n=3,
        similarity_top_k=5,
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        llm_model="gemini-3.5-flash-lite",
        chunk_size=200,
        chunk_overlap=20,
    )

    service = RAGService(settings)
    result = asyncio.run(service.ingest(source_dir))

    assert result.documents >= 1
    assert result.nodes >= 1
    assert service.ready is True
    assert service.indexes is not None
    assert service.index_factory.manifest_path.exists()

    restored = service.index_factory.restore()
    assert restored is not None
    restored_indexes, metadata = restored
    assert restored_indexes.vector is not None
    assert restored_indexes.summary is not None
    assert int(metadata["documents"]) == result.documents
    assert int(metadata["nodes"]) == result.nodes
