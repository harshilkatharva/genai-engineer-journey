"""Unit tests for SourceRetriever."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from genai_core import DocumentChunk, RetrievalResponse, RetrievalResult

from self_improving_report_writer.core.exceptions import RetrievalError
from self_improving_report_writer.retrieval.source_retriever import SourceRetriever


@pytest.fixture
def mock_retrieval_service() -> AsyncMock:
    """Mock genai-core RetrievalService."""
    service = AsyncMock()
    service.search = AsyncMock()
    service.index_documents = AsyncMock(return_value=2)
    return service


async def test_retrieve_sources_success(mock_retrieval_service: AsyncMock) -> None:
    """Retrieve sources should map RetrievalResult objects into SourceReferences."""
    mock_retrieval_service.search.return_value = RetrievalResponse(
        results=[
            RetrievalResult(
                id="chunk_1",
                content="Quantum speedup content",
                score=0.92,
                rank=1,
                source_id="src_1",
                metadata={"title": "Quantum Paper"},
            ),
            RetrievalResult(
                id="chunk_2",
                content="Classical baseline content",
                score=0.85,
                rank=2,
                source_id="src_2",
                metadata={"title": "Classical Paper"},
            ),
        ],
        total_results=2,
        latency_ms=12.5,
    )

    retriever = SourceRetriever(retrieval_service=mock_retrieval_service)
    sources = await retriever.retrieve_sources(topic="Quantum Optimization", top_k=5)

    assert len(sources) == 2
    assert sources[0].source_id == "src_1"
    assert sources[0].chunk_id == "chunk_1"
    assert sources[0].title == "Quantum Paper"
    assert sources[0].content == "Quantum speedup content"
    assert sources[0].score == 0.92

    assert sources[1].source_id == "src_2"
    assert sources[1].chunk_id == "chunk_2"
    assert sources[1].score == 0.85


async def test_retrieve_sources_min_score_filter(mock_retrieval_service: AsyncMock) -> None:
    """Chunks below min_score should be filtered out."""
    mock_retrieval_service.search.return_value = RetrievalResponse(
        results=[
            RetrievalResult(
                id="c1",
                content="High score",
                score=0.85,
                rank=1,
                source_id="s1",
            ),
            RetrievalResult(
                id="c2",
                content="Low score",
                score=0.35,
                rank=2,
                source_id="s2",
            ),
        ],
        total_results=2,
        latency_ms=10.0,
    )

    retriever = SourceRetriever(retrieval_service=mock_retrieval_service)
    sources = await retriever.retrieve_sources(topic="Topic", min_score=0.50)

    assert len(sources) == 1
    assert sources[0].chunk_id == "c1"


async def test_retrieve_sources_empty_topic_raises() -> None:
    """Empty or whitespace-only topic should raise RetrievalError."""
    retriever = SourceRetriever()
    with pytest.raises(RetrievalError, match="Topic cannot be empty"):
        await retriever.retrieve_sources(topic="   ")


async def test_retrieve_sources_service_error_handling(
    mock_retrieval_service: AsyncMock,
) -> None:
    """Underlying service exceptions should be wrapped in RetrievalError."""
    mock_retrieval_service.search.side_effect = RuntimeError("Database timeout")

    retriever = SourceRetriever(retrieval_service=mock_retrieval_service)
    with pytest.raises(RetrievalError, match="Evidence retrieval failed"):
        await retriever.retrieve_sources(topic="Faulty Query")


async def test_index_sources_convenience(mock_retrieval_service: AsyncMock) -> None:
    """Index sources should convert inputs and invoke index_documents."""
    retriever = SourceRetriever(retrieval_service=mock_retrieval_service)
    docs = [
        DocumentChunk(id="c1", content="Chunk 1", source_id="s1"),
        {"chunk_id": "c2", "content": "Chunk 2", "source_id": "s2"},
    ]
    count = await retriever.index_sources(docs)
    assert count == 2
    mock_retrieval_service.index_documents.assert_called_once()
