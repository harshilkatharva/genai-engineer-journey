"""Integration test for LLMService structured generation and live Gemini API connectivity."""

from __future__ import annotations

import os

import pytest
from genai_core.services.llm_services import LLMService

from parallel_research_team.core.config import ResearchTeamConfig
from parallel_research_team.models.decomposition import EffortLevel, ResearchDecomposition
from parallel_research_team.models.report import Report


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_service_decomposition_structured_output() -> None:
    """Verify live LLM connectivity and structured completion for ResearchDecomposition schema."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY is not set.")

    config = ResearchTeamConfig()
    llm_service = LLMService(
        default_provider=config.default_provider,
        default_model=config.default_model,
    )

    # Use mock source guidelines and prompt data to test pure LLM connectivity & schema adherence
    mock_prompt = """
You are a Lead Research Agent. Decompose the following research topic into bounded worker briefs:
Topic: "Comparative analysis of HNSW vs DiskANN indexing for large-scale RAG"

Instructions:
1. Classify the effort level as either DIRECT or PARALLEL_DECOMPOSED.
2. Produce 2 orthogonal worker briefs with clear boundaries and objectives.
3. Include suggested tools and maximum step counts.
Respond strictly matching the requested schema.
"""

    response = await llm_service.complete_structure(
        prompt=mock_prompt,
        response_schema=ResearchDecomposition,
        provider=config.default_provider,
        model=config.default_model,
    )

    assert response.error is None, f"LLM returned error: {response.error}"
    assert response.structured_response is not None, "Expected structured_response from LLM"
    assert isinstance(response.structured_response, ResearchDecomposition)

    decomposition: ResearchDecomposition = response.structured_response
    assert decomposition.topic != ""
    assert decomposition.effort_level in (
        EffortLevel.DIRECT,
        EffortLevel.PARALLEL_DECOMPOSED,
    )
    assert len(decomposition.worker_briefs) >= 1

    for brief in decomposition.worker_briefs:
        assert brief.brief_id != ""
        assert brief.title != ""
        assert brief.objective != ""
        assert len(brief.task_boundaries) >= 1

    # Verify real usage telemetry (dynamic tokens)
    assert response.usage is not None
    assert response.usage.get("input_tokens", 0) > 0
    assert response.usage.get("output_tokens", 0) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_service_report_synthesis_structured_output() -> None:
    """Verify live LLM connectivity and structured synthesis into Report schema with mock sources."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY is not set.")

    config = ResearchTeamConfig()
    llm_service = LLMService(
        default_provider=config.default_provider,
        default_model=config.default_model,
    )

    # Mock evidence pool and worker findings for synthesis testing
    mock_synthesis_prompt = """
You are a Lead Research Synthesizer. Synthesize a comprehensive research report from the findings below:

Topic: "Vector Search Memory Optimization"

Worker Brief 1 Findings:
"HNSW index stores vectors in RAM with multi-layer graphs, achieving 5ms latency at 1.5x RAM overhead."
Sources:
- Source ID: src_001, Chunk ID: chk_001, Title: "HNSW Deep Dive", Quote: "HNSW requires 1.5x RAM overhead for fast graph traversal."

Worker Brief 2 Findings:
"DiskANN compresses vectors onto SSD using Vamana graphs, reducing DRAM requirements by 80% with 15ms latency."
Sources:
- Source ID: src_002, Chunk ID: chk_002, Title: "DiskANN Paper", Quote: "Vamana graphs enable SSD-resident search with modest DRAM."

Synthesize the report adhering strictly to the Report schema. Include executive summary, sections with claims and citations, and conclusion.
"""

    response = await llm_service.complete_structure(
        prompt=mock_synthesis_prompt,
        response_schema=Report,
        provider=config.default_provider,
        model=config.default_model,
    )

    assert response.error is None, f"LLM returned error: {response.error}"
    assert response.structured_response is not None, "Expected structured_response from LLM"
    assert isinstance(response.structured_response, Report)

    report: Report = response.structured_response
    assert report.topic != ""
    assert report.title != ""
    assert len(report.executive_summary.content) > 20
    assert len(report.sections) >= 1

    first_section = report.sections[0]
    assert first_section.title != ""
    assert len(first_section.content) > 10

    # Verify claims and citations if present
    if first_section.claims:
        for claim in first_section.claims:
            assert claim.text != ""
            for cit in claim.citations:
                assert cit.source_id != ""
                assert cit.chunk_id != ""

    # Verify real usage telemetry
    assert response.usage is not None
    assert response.usage.get("input_tokens", 0) > 0
    assert response.usage.get("output_tokens", 0) > 0
