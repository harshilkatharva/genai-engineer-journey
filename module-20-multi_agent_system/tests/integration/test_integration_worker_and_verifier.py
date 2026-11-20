"""Integration tests for Module 17 bounded research worker and Module 19 reflection verifiers."""

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
from genai_core.services.llm_services import LLMService
from genai_core.services.retrieval_service import RetrievalService

from parallel_research_team.core.budget import SharedRunBudget
from parallel_research_team.core.config import ResearchTeamConfig
from parallel_research_team.models.brief import BriefType, OutputFormatSpec, WorkerBrief
from parallel_research_team.models.report import (
    BriefCoverage,
    Citation,
    CitationType,
    Claim,
    Report,
    ReportSection,
    SourceReference,
)
from parallel_research_team.models.worker import WorkerStatus
from parallel_research_team.synthesis.source_pool import AuthoritativeSourcePool
from parallel_research_team.synthesis.verifier import ResearchVerifier
from parallel_research_team.workers.worker import DefaultResearchWorker

FIXTURES_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "research_chunks.json"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_bounded_research_worker_execution() -> None:
    """Verify DefaultResearchWorker execution with live Gemini LLM and retrieval tools."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY is not set.")

    if not FIXTURES_PATH.exists():
        pytest.skip(f"Fixture file not found: {FIXTURES_PATH}")

    # 1. Setup in-memory knowledge base with real handbook chunks
    content = await asyncio.to_thread(FIXTURES_PATH.read_text, encoding="utf-8")
    raw_chunks: list[dict[str, Any]] = json.loads(content)[:50]

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
    retrieval_service = RetrievalService(store=store, embedding_service=emb_service)

    # 2. Setup config, budget, and live LLM
    config = ResearchTeamConfig(
        default_model="gemini-3.5-flash-lite",
        per_worker_max_steps=3,
        per_worker_timeout_seconds=30.0,
    )
    budget = SharedRunBudget(config=config, max_total_tokens=20_000, max_total_cost_usd=0.20)
    llm_service = LLMService(
        default_provider=config.default_provider,
        default_model=config.default_model,
    )

    worker = DefaultResearchWorker(
        config=config,
        llm_service=llm_service,
        retrieval_service=retrieval_service,
    )

    brief = WorkerBrief(
        brief_id="brief_worker_test",
        title="Production GenAI Handbook Overview",
        topic="Production GenAI and Agentic AI Architecture",
        objective="Find out what topics and modules the Production GenAI handbook covers using the knowledge base.",
        brief_type=BriefType.FACTUAL_RETRIEVAL,
        task_boundaries=[
            "Search the knowledge base for handbook scope and modules.",
            "Do NOT investigate external unrelated websites.",
        ],
        expected_output_format=OutputFormatSpec(
            schema_name="WorkerResearchOutput",
            required_fields=["findings_summary", "claims", "sources"],
            guidelines="Summarize the handbook topics with cited source IDs.",
        ),
        suggested_tools=["search_knowledge_base"],
        max_steps=3,
        timeout_seconds=30.0,
    )

    result = await worker.execute(
        brief=brief,
        shared_budget=budget,
    )

    assert result.status in (WorkerStatus.COMPLETED, WorkerStatus.PARTIAL)
    assert result.duration_ms > 0
    assert result.output is not None
    assert len(result.output.findings_summary) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_research_verifier_two_stage_gates_live() -> None:
    """Verify ResearchVerifier Gate 1 (deterministic) and Gate 2 (live semantic judge)."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY is not set.")

    config = ResearchTeamConfig(
        judge_provider="google",
        judge_model="gemini-3.1-flash-lite",
        enable_semantic_verification=True,
        semantic_verification_threshold=0.70,
    )

    judge_llm_service = LLMService(
        default_provider=config.judge_provider,
        default_model=config.judge_model,
    )

    pool = AuthoritativeSourcePool()
    real_chunk_text = (
        "The Production-Ready GenAI and Agentic AI Engineer Handbook covers "
        "30 modules across core LLM fundamentals, RAG, planning, and multi-agent systems."
    )
    source_ref = SourceReference(
        source_id="src_handbook_01",
        chunk_id="chk_handbook_01",
        title="Production GenAI Handbook",
        content=real_chunk_text,
        score=0.95,
        metadata={"author": "Atharva System"},
    )
    pool.add_source(source_ref)

    # Construct structurally valid report with verifiable quotation citation
    quote_text = "The Production-Ready GenAI and Agentic AI Engineer Handbook covers 30 modules"
    cite_01 = Citation(
        citation_id="cite_01",
        source_id="src_handbook_01",
        chunk_id="chk_handbook_01",
        citation_type=CitationType.QUOTATION,
        quoted_passage=quote_text,
        confidence=0.98,
    )
    cite_02 = Citation(
        citation_id="cite_02",
        source_id="src_handbook_01",
        chunk_id="chk_handbook_01",
        citation_type=CitationType.QUOTATION,
        quoted_passage=quote_text,
        confidence=0.98,
    )
    cite_03 = Citation(
        citation_id="cite_03",
        source_id="src_handbook_01",
        chunk_id="chk_handbook_01",
        citation_type=CitationType.QUOTATION,
        quoted_passage=quote_text,
        confidence=0.98,
    )
    claim = Claim(
        claim_id="claim_01",
        section_id="sec_exec",
        text="The Production-Ready GenAI and Agentic AI Engineer Handbook covers 30 modules across core LLM fundamentals, RAG, planning, and multi-agent systems.",
        citations=[cite_01],
        is_factual=True,
    )
    sec_exec = ReportSection(
        section_id="sec_exec",
        title="Executive Summary",
        content="The Production-Ready GenAI and Agentic AI Engineer Handbook covers 30 modules across core LLM fundamentals, RAG, planning, and multi-agent systems [cite_01].",
        claims=[claim],
    )
    sec_body = ReportSection(
        section_id="sec_body",
        title="Curriculum Details",
        content="As detailed in the executive summary, all curriculum topics are organized into 30 modules [cite_02].",
        claims=[
            Claim(
                claim_id="claim_02",
                section_id="sec_body",
                text="All curriculum topics are organized into 30 modules.",
                citations=[cite_02],
                is_factual=True,
            )
        ],
    )
    sec_concl = ReportSection(
        section_id="sec_concl",
        title="Conclusion",
        content="This concludes the curriculum overview of the 30 modules [cite_03].",
        claims=[
            Claim(
                claim_id="claim_03",
                section_id="sec_concl",
                text="The curriculum covers 30 modules.",
                citations=[cite_03],
                is_factual=True,
            )
        ],
    )

    report = Report(
        topic="Production GenAI Curriculum",
        title="Curriculum Analysis of GenAI Engineer Handbook",
        executive_summary=sec_exec,
        sections=[sec_body],
        conclusion=sec_concl,
        limitations=[],
        brief_coverages=[
            BriefCoverage(brief_id="brief_01", coverage_summary="Curriculum scope analyzed")
        ],
    )

    verifier = ResearchVerifier(
        config=config,
        judge_llm_service=judge_llm_service,
    )

    verification_result = await verifier.verify(report=report, source_pool=pool)

    # Gate 1 structural & deterministic verification must pass
    assert verification_result.gate1_passed is True

    # Gate 2 semantic grounding evaluation
    assert verification_result.gate2_passed is not None
    assert verification_result.passed is True
    assert (
        verification_result.gate2_report is not None
        and verification_result.gate2_report.grounding_score >= 0.70
    )
