"""Unit tests for research tools, evidence collector, and evidence extractor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from genai_core.models.retrieval_model import RetrievalResponse, RetrievalResult
from genai_core.retrieval.in_memory_store import InMemoryVectorStore

from parallel_research_team.core.config import ResearchTeamConfig
from parallel_research_team.exceptions import WorkerExecutionError
from parallel_research_team.models import WorkerBrief
from parallel_research_team.workers.extractor import (
    EvidenceCollector,
    EvidenceExtractor,
    create_deterministic_chunk_id,
    create_deterministic_source_id,
)
from parallel_research_team.workers.tools import (
    create_research_tool_registry,
    create_retrieval_service,
)


def test_deterministic_id_generation() -> None:
    url = "https://example.com/research/ai-agents"
    src_id = create_deterministic_source_id(url, prefix="web")
    assert src_id.startswith("web_")
    assert len(src_id) == 16  # "web_" + 12 chars
    # Determinism: same input gives identical id
    assert src_id == create_deterministic_source_id(url, prefix="web")

    content = "Multi-agent systems enable concurrent task execution."
    chunk_id = create_deterministic_chunk_id(src_id, content)
    assert chunk_id.startswith(f"{src_id}_chunk_")
    assert chunk_id == create_deterministic_chunk_id(src_id, content)


def test_evidence_collector() -> None:
    collector = EvidenceCollector()
    assert not collector.has_sources()

    collector.record_query("ai agent planning")
    collector.record_query("ai agent planning")  # duplicate deduplication
    assert collector.get_queries() == ["ai agent planning"]

    collector.record_error("Failed to reach external service")
    assert collector.get_errors() == ["Failed to reach external service"]

    ref = collector.add_web_source(
        url="https://example.com/agents",
        title="Agent Architecture",
        content="Agents break complex goals into tasks.",
    )
    assert collector.has_sources()
    assert ref.title == "Agent Architecture"
    assert ref.metadata["url"] == "https://example.com/agents"
    assert len(collector.get_sources()) == 1


def test_create_retrieval_service_in_memory() -> None:
    config = ResearchTeamConfig(retrieval_mode="in_memory")
    service = create_retrieval_service(config=config)
    assert isinstance(service.store, InMemoryVectorStore)


def test_create_retrieval_service_postgres_missing_dsn_raises() -> None:
    config = ResearchTeamConfig(retrieval_mode="postgres", postgres_connection_string="")
    with pytest.raises(WorkerExecutionError, match="requires 'postgres_connection_string'"):
        create_retrieval_service(config=config)


def test_create_retrieval_service_invalid_mode_raises() -> None:
    config = ResearchTeamConfig()
    config.retrieval_mode = "unsupported_mode"  # type: ignore[assignment]
    with pytest.raises(WorkerExecutionError, match="Unsupported retrieval mode"):
        create_retrieval_service(config=config)


@pytest.mark.asyncio
async def test_research_tools_execution() -> None:
    collector = EvidenceCollector()

    # 1. Mock HTTP client for search_web (Tavily JSON API)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {
                "title": "Doc Title 1",
                "url": "https://example.com/doc1",
                "content": "Snippet 1 details on parallel research.",
                "score": 0.95,
            }
        ]
    }

    mock_http_client = AsyncMock(spec=httpx.AsyncClient)
    mock_http_client.post.return_value = mock_response

    # 2. Mock retrieval service for search_knowledge_base
    mock_retrieval_service = AsyncMock()
    mock_retrieval_service.search.return_value = RetrievalResponse(
        results=[
            RetrievalResult(
                id="chunk_kb_1",
                source_id="kb_source_1",
                content="Internal policy for parallel multi-agent dispatch.",
                score=0.92,
                rank=1,
                metadata={"title": "Dispatch Policy"},
            )
        ],
        total_results=1,
        latency_ms=10.5,
    )

    tool_registry = create_research_tool_registry(
        collector=collector,
        retrieval_service=mock_retrieval_service,
        http_client=mock_http_client,
    )

    assert tool_registry.has("search_web")
    assert tool_registry.has("search_knowledge_base")

    # Run search_web
    web_tool = tool_registry.get_or_fail("search_web")
    web_out = await web_tool.run(query="multi agent research", max_results=1)
    assert "Doc Title 1" in web_out
    assert "Snippet 1 details" in web_out
    assert len(collector.get_sources()) == 1

    # Run search_knowledge_base
    kb_tool = tool_registry.get_or_fail("search_knowledge_base")
    kb_out = await kb_tool.run(query="dispatch policy", top_k=1)
    assert "Dispatch Policy" in kb_out
    assert "Internal policy for parallel" in kb_out
    assert len(collector.get_sources()) == 2


def test_evidence_extractor() -> None:
    collector = EvidenceCollector()
    ref = collector.add_web_source(
        url="https://example.com/agents",
        title="Agent Architecture",
        content="Agents break complex goals into tasks.",
    )
    collector.record_query("agent architecture")

    brief = WorkerBrief(
        brief_id="brief_01",
        title="Architecture Brief",
        topic="Multi-Agent",
        objective="Analyze agent architecture",
        task_boundaries=["Focus on architecture", "Do not write code"],
    )

    extractor = EvidenceExtractor()
    output = extractor.extract(
        brief=brief,
        final_text="Agent systems use decomposition. Agents break complex goals into tasks.",
        collector=collector,
    )

    assert output.brief_id == "brief_01"
    assert len(output.sources) == 1
    assert output.sources[0]["source_id"] == ref.source_id
    assert output.search_queries_used == ["agent architecture"]
    assert len(output.claims) >= 1
    assert output.claims[0]["citations"][0]["source_id"] == ref.source_id


def test_evidence_collector_empty_content_graceful() -> None:
    collector = EvidenceCollector()
    ref = collector.add_web_source(
        url="https://example.com/empty",
        title="Empty Page",
        content="",
    )
    assert ref.source_id.startswith("web_")
    assert ref.content == "Empty Page"
    assert len(collector.get_sources()) == 1


@pytest.mark.asyncio
async def test_research_tools_empty_snippet_graceful() -> None:
    from unittest.mock import AsyncMock, MagicMock

    import httpx

    from parallel_research_team.workers.tools import create_research_tool_registry

    collector = EvidenceCollector()
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {"title": "Result 1", "url": "https://example.com/1", "content": ""},
            {
                "title": "Result 2",
                "url": "https://example.com/2",
                "content": "Valid snippet content.",
            },
        ]
    }
    mock_http_client = AsyncMock(spec=httpx.AsyncClient)
    mock_http_client.post.return_value = mock_response

    registry = create_research_tool_registry(collector=collector, http_client=mock_http_client)

    tool = registry.get_or_fail("search_web")
    output = await tool.run(query="test query")

    assert "Result 1" in output
    assert "Result 2" in output
    assert len(collector.get_sources()) == 2
