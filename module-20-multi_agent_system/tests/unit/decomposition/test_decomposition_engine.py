"""Unit tests for DecompositionEngine."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from parallel_research_team.core.config import ResearchTeamConfig
from parallel_research_team.decomposition.engine import DecompositionEngine
from parallel_research_team.exceptions import DecompositionError
from parallel_research_team.models import (
    BriefType,
    EffortLevel,
    OutputFormatSpec,
    ResearchDecomposition,
    WorkerBrief,
)


def _create_mock_response(plan: ResearchDecomposition) -> MagicMock:
    resp = MagicMock()
    resp.error = None
    resp.structured_response = plan
    resp.usage = {"input_tokens": 120, "output_tokens": 250}
    return resp


@pytest.mark.asyncio
async def test_decomposition_direct_simple_question(test_config: ResearchTeamConfig) -> None:
    brief = WorkerBrief(
        brief_id="brief_01",
        title="Python GIL Explanation",
        topic="What is the Python GIL?",
        objective="Explain the mechanics of the Global Interpreter Lock",
        brief_type=BriefType.FACTUAL_RETRIEVAL,
        task_boundaries=["Focus on GIL in CPython", "Do NOT research free-threading"],
        expected_output_format=OutputFormatSpec(),
    )
    direct_plan = ResearchDecomposition(
        topic="What is the Python GIL?",
        effort_level=EffortLevel.DIRECT,
        rationale="Simple factual concept lookup",
        worker_count=1,
        worker_briefs=[brief],
    )

    mock_llm = MagicMock()
    mock_llm.complete_structure = AsyncMock(return_value=_create_mock_response(direct_plan))

    engine = DecompositionEngine(llm_service=mock_llm, config=test_config)
    result = await engine.decompose(topic="What is the Python GIL?")

    assert result.effort_level == EffortLevel.DIRECT
    assert result.worker_count == 1
    assert len(result.worker_briefs) == 1
    assert result.worker_briefs[0].brief_id == "brief_01"
    mock_llm.complete_structure.assert_awaited_once()


@pytest.mark.asyncio
async def test_decomposition_parallel_broad_question(test_config: ResearchTeamConfig) -> None:
    b1 = WorkerBrief(
        brief_id="brief_01",
        title="Postgres Indexing",
        topic="Postgres vs DynamoDB",
        objective="Examine Postgres B-Tree and GIN indexing",
        brief_type=BriefType.COMPARATIVE_ANALYSIS,
        task_boundaries=["Focus on Postgres", "Do NOT research DynamoDB"],
        expected_output_format=OutputFormatSpec(),
    )
    b2 = WorkerBrief(
        brief_id="brief_02",
        title="DynamoDB Partitioning",
        topic="Postgres vs DynamoDB",
        objective="Examine DynamoDB partition keys and global secondary indexes",
        brief_type=BriefType.COMPARATIVE_ANALYSIS,
        task_boundaries=["Focus on DynamoDB", "Do NOT research Postgres"],
        expected_output_format=OutputFormatSpec(),
    )
    parallel_plan = ResearchDecomposition(
        topic="Postgres vs DynamoDB",
        effort_level=EffortLevel.PARALLEL_DECOMPOSED,
        rationale="Multi-database comparative investigation",
        worker_count=2,
        worker_briefs=[b1, b2],
    )

    mock_llm = MagicMock()
    mock_llm.complete_structure = AsyncMock(return_value=_create_mock_response(parallel_plan))

    engine = DecompositionEngine(llm_service=mock_llm, config=test_config)
    result = await engine.decompose(topic="Postgres vs DynamoDB")

    assert result.effort_level == EffortLevel.PARALLEL_DECOMPOSED
    assert result.worker_count == 2
    assert len(result.worker_briefs) == 2


@pytest.mark.asyncio
async def test_decomposition_memory_integration(test_config: ResearchTeamConfig) -> None:
    brief = WorkerBrief(
        brief_id="brief_01",
        title="Indexing Research",
        topic="Vector Indexing",
        objective="Research indexing algorithms",
        brief_type=BriefType.EXPLORATORY_DEEP_DIVE,
        task_boundaries=["Focus on algorithms", "Do NOT cover storage hardware"],
        expected_output_format=OutputFormatSpec(),
    )
    plan = ResearchDecomposition(
        topic="Vector Indexing",
        effort_level=EffortLevel.DIRECT,
        rationale="Focused research",
        worker_count=1,
        worker_briefs=[brief],
    )

    mock_llm = MagicMock()
    mock_llm.complete_structure = AsyncMock(return_value=_create_mock_response(plan))

    mock_memory = MagicMock()
    # Mock recalling preferences from LTM
    mock_recall_result = MagicMock()
    mock_recall_result.content = "User prefers memory-efficient algorithms"
    mock_memory.recall = AsyncMock(return_value=[mock_recall_result])
    mock_memory.pin_goal = AsyncMock()

    engine = DecompositionEngine(
        llm_service=mock_llm, config=test_config, memory_service=mock_memory
    )
    result = await engine.decompose(
        topic="Vector Indexing",
        user_id="user_test_123",
        session_id="session_test_456",
    )

    assert result.worker_count == 1
    mock_memory.recall.assert_awaited_once_with(
        user_id="user_test_123", query="Vector Indexing", top_k=3
    )
    mock_memory.pin_goal.assert_awaited_once()


@pytest.mark.asyncio
async def test_decomposition_empty_topic_raises(test_config: ResearchTeamConfig) -> None:
    mock_llm = MagicMock()
    engine = DecompositionEngine(llm_service=mock_llm, config=test_config)
    with pytest.raises(DecompositionError, match="Cannot decompose an empty"):
        await engine.decompose(topic="   ")


@pytest.mark.asyncio
async def test_decomposition_llm_failure_raises(test_config: ResearchTeamConfig) -> None:
    mock_llm = MagicMock()
    mock_llm.complete_structure = AsyncMock(side_effect=RuntimeError("API Network Timeout"))

    engine = DecompositionEngine(llm_service=mock_llm, config=test_config)
    with pytest.raises(DecompositionError, match="LLM decomposition completion call failed"):
        await engine.decompose(topic="Any Topic")


@pytest.mark.asyncio
async def test_decomposition_non_schema_response_raises(test_config: ResearchTeamConfig) -> None:
    mock_llm = MagicMock()
    resp = MagicMock()
    resp.error = None
    resp.structured_response = "Not a ResearchDecomposition"
    mock_llm.complete_structure = AsyncMock(return_value=resp)

    engine = DecompositionEngine(llm_service=mock_llm, config=test_config)
    with pytest.raises(DecompositionError, match="failed to return a valid ResearchDecomposition"):
        await engine.decompose(topic="Any Topic")
