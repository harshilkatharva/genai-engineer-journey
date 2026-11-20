"""Integration tests comparing Single-Agent (DIRECT) vs Multi-Agent (PARALLEL_DECOMPOSED) execution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from genai_core.embedding.local_provider import LocalEmbeddingProvider
from genai_core.models import DocumentChunk
from genai_core.retrieval.in_memory_store import InMemoryVectorStore
from genai_core.services.embedding_service import EmbeddingService
from genai_core.services.llm_services import LLMService
from genai_core.services.retrieval_service import RetrievalService

from parallel_research_team.core.config import ResearchTeamConfig
from parallel_research_team.models.checkpoint import RunStatus
from parallel_research_team.models.decomposition import EffortLevel
from parallel_research_team.orchestrator import ParallelResearchTeam
from parallel_research_team.storage.file_store import FileCheckpointStore

FIXTURES_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "research_chunks.json"


def _create_test_retrieval_service() -> RetrievalService:
    """Build in-memory retrieval service loaded with real handbook chunks."""
    with open(FIXTURES_PATH) as f:
        raw_chunks: list[dict[str, Any]] = json.load(f)[:50]

    document_chunks = [
        DocumentChunk(
            id=c["chunk_id"],
            content=c["chunk_text"],
            embedding=json.loads(c["embedding"])
            if isinstance(c["embedding"], str)
            else c["embedding"],
            metadata=c.get("metadata") or {},
            source_id=c.get("metadata", {}).get("document_name", "handbook"),
        )
        for c in raw_chunks
    ]
    store = InMemoryVectorStore(dimensions=384, chunks=document_chunks)
    emb_service = EmbeddingService(provider=LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2"))
    return RetrievalService(store=store, embedding_service=emb_service)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_orchestrator_single_agent_direct_mode(tmp_path: Path) -> None:
    """Verify single-agent DIRECT execution mode with live Gemini LLM."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY is not set.")

    if not FIXTURES_PATH.exists():
        pytest.skip(f"Fixture file not found: {FIXTURES_PATH}")

    config = ResearchTeamConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        default_model="gemini-3.5-flash-lite",
        judge_model="gemini-3.1-flash-lite",
        max_workers=1,
        per_worker_max_steps=4,
        per_worker_timeout_seconds=30.0,
        enable_semantic_verification=False,  # Focus on orchestrator flow & budget telemetry
    )

    llm_service = LLMService(
        default_provider=config.default_provider,
        default_model=config.default_model,
    )
    retrieval_service = _create_test_retrieval_service()

    team = ParallelResearchTeam(
        config=config,
        llm_service=llm_service,
        retrieval_service=retrieval_service,
    )

    topic = "What is the scope and audience of the Production GenAI Handbook?"
    result = await team.run(
        topic=topic,
        effort_override=EffortLevel.DIRECT,
    )

    assert result.status in (RunStatus.COMPLETED, RunStatus.PARTIALLY_COMPLETED)
    assert result.report is not None
    assert result.report.topic == topic
    assert len(result.report.sections) >= 1

    # Single-agent assertions
    assert result.metrics.total_workers == 1
    assert result.metrics.total_tokens > 0
    assert result.metrics.total_cost_usd > 0.0
    assert result.metrics.duration_ms > 0.0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_orchestrator_multi_agent_parallel_decomposed_mode(tmp_path: Path) -> None:
    """Verify multi-agent PARALLEL_DECOMPOSED execution mode with live Gemini LLM."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY is not set.")

    if not FIXTURES_PATH.exists():
        pytest.skip(f"Fixture file not found: {FIXTURES_PATH}")

    config = ResearchTeamConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        default_model="gemini-3.5-flash-lite",
        judge_model="gemini-3.1-flash-lite",
        max_workers=2,
        max_concurrency=2,
        per_worker_max_steps=4,
        per_worker_timeout_seconds=30.0,
        enable_semantic_verification=False,
    )

    llm_service = LLMService(
        default_provider=config.default_provider,
        default_model=config.default_model,
    )
    retrieval_service = _create_test_retrieval_service()

    team = ParallelResearchTeam(
        config=config,
        llm_service=llm_service,
        retrieval_service=retrieval_service,
    )

    topic = "Compare RAG indexing architectures and evaluation methods in production GenAI"
    result = await team.run(
        topic=topic,
        effort_override=EffortLevel.PARALLEL_DECOMPOSED,
    )

    assert result.status in (RunStatus.COMPLETED, RunStatus.PARTIALLY_COMPLETED)
    assert result.report is not None
    assert result.report.topic == topic
    assert len(result.report.sections) >= 1

    # Multi-agent assertions
    assert result.metrics.total_workers >= 2
    assert result.metrics.total_tokens > 0
    assert result.metrics.total_cost_usd > 0.0
    assert result.metrics.duration_ms > 0.0

    # Verify checkpoint persistence on disk
    checkpoint_store = FileCheckpointStore(config=config)
    persisted = await checkpoint_store.get_checkpoint(result.run_id)
    assert persisted is not None
    assert persisted.status in (RunStatus.COMPLETED, RunStatus.PARTIALLY_COMPLETED)
    assert persisted.decomposition is not None
    assert len(persisted.worker_results) >= 2
