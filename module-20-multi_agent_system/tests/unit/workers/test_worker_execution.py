"""Unit tests for DefaultResearchWorker execution, timeouts, boundaries, and failure isolation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genai_core import ChatMessage, ToolLoopResult, ToolLoopStatus

from parallel_research_team.core.budget import SharedRunBudget
from parallel_research_team.core.config import ResearchTeamConfig
from parallel_research_team.models import (
    WorkerBrief,
    WorkerStatus,
)
from parallel_research_team.prompts.manager import PromptTemplateManager
from parallel_research_team.workers.worker import DefaultResearchWorker


@pytest.fixture
def sample_brief() -> WorkerBrief:
    return WorkerBrief(
        brief_id="brief_test_1",
        title="Agent Concurrency",
        topic="Multi-Agent Systems",
        objective="Investigate concurrency control mechanisms",
        task_boundaries=["Focus on locking mechanisms", "Exclude legacy threads"],
        max_steps=3,
        timeout_seconds=5.0,
    )


@pytest.mark.asyncio
async def test_worker_successful_execution(
    test_config: ResearchTeamConfig, sample_brief: WorkerBrief
) -> None:
    budget = SharedRunBudget(max_total_tokens=10000, max_total_cost_usd=1.0)
    mock_llm = MagicMock()

    worker = DefaultResearchWorker(
        config=test_config,
        llm_service=mock_llm,
        prompt_manager=PromptTemplateManager(config=test_config),
    )

    mock_loop_result = ToolLoopResult(
        status=ToolLoopStatus.COMPLETED,
        final_text="Concurrency control requires atomic locking.",
        messages=[
            ChatMessage(role="user", content="Research task"),
            ChatMessage(
                role="assistant",
                content="Concurrency control requires atomic locking.",
            ),
        ],
        iterations=2,
        total_tool_calls=1,
    )

    with patch(
        "parallel_research_team.workers.worker.ToolLoopRunner.run",
        new_callable=AsyncMock,
    ) as mock_run:
        mock_run.return_value = mock_loop_result
        result = await worker.execute(brief=sample_brief, shared_budget=budget)

    assert result.brief_id == sample_brief.brief_id
    assert result.status == WorkerStatus.COMPLETED
    assert result.output is not None
    assert "Concurrency control requires" in result.output.findings_summary
    assert result.error_message is None


@pytest.mark.asyncio
async def test_worker_step_limit_partial_status(
    test_config: ResearchTeamConfig, sample_brief: WorkerBrief
) -> None:
    budget = SharedRunBudget(max_total_tokens=10000, max_total_cost_usd=1.0)
    mock_llm = MagicMock()

    worker = DefaultResearchWorker(
        config=test_config,
        llm_service=mock_llm,
        prompt_manager=PromptTemplateManager(config=test_config),
    )

    mock_loop_result = ToolLoopResult(
        status=ToolLoopStatus.MAX_ITERATIONS_REACHED,
        final_text="Partial findings before hitting step ceiling.",
        messages=[],
        iterations=3,
    )

    with patch(
        "parallel_research_team.workers.worker.ToolLoopRunner.run",
        new_callable=AsyncMock,
    ) as mock_run:
        mock_run.return_value = mock_loop_result
        result = await worker.execute(brief=sample_brief, shared_budget=budget)

    assert result.status == WorkerStatus.PARTIAL
    assert result.output is not None
    assert "Partial findings" in result.output.findings_summary
    assert "Step limit (3) reached" in (result.error_message or "")


@pytest.mark.asyncio
async def test_worker_timeout_isolation_and_salvage(
    test_config: ResearchTeamConfig, sample_brief: WorkerBrief
) -> None:
    budget = SharedRunBudget(max_total_tokens=10000, max_total_cost_usd=1.0)
    mock_llm = MagicMock()

    worker = DefaultResearchWorker(
        config=test_config,
        llm_service=mock_llm,
        prompt_manager=PromptTemplateManager(config=test_config),
    )

    with patch(
        "parallel_research_team.workers.worker.ToolLoopRunner.run",
        new_callable=AsyncMock,
    ) as mock_run:
        mock_run.side_effect = TimeoutError()
        result = await worker.execute(brief=sample_brief, shared_budget=budget)

    assert result.status == WorkerStatus.TIMEOUT
    assert "timed out" in (result.error_message or "").lower()


@pytest.mark.asyncio
async def test_worker_pre_exhausted_budget(
    test_config: ResearchTeamConfig, sample_brief: WorkerBrief
) -> None:
    budget = SharedRunBudget(max_total_tokens=100, max_total_cost_usd=1.0)
    await budget.record_usage(tokens=150, cost_usd=0.01)

    mock_llm = MagicMock()
    worker = DefaultResearchWorker(
        config=test_config,
        llm_service=mock_llm,
        prompt_manager=PromptTemplateManager(config=test_config),
    )

    result = await worker.execute(brief=sample_brief, shared_budget=budget)
    assert result.status == WorkerStatus.FAILED
    assert "Shared run budget already exhausted" in (result.error_message or "")


@pytest.mark.asyncio
async def test_worker_unhandled_exception_shielding(
    test_config: ResearchTeamConfig, sample_brief: WorkerBrief
) -> None:
    budget = SharedRunBudget(max_total_tokens=10000, max_total_cost_usd=1.0)
    mock_llm = MagicMock()

    worker = DefaultResearchWorker(
        config=test_config,
        llm_service=mock_llm,
        prompt_manager=PromptTemplateManager(config=test_config),
    )

    with patch(
        "parallel_research_team.workers.worker.ToolLoopRunner.run",
        new_callable=AsyncMock,
    ) as mock_run:
        mock_run.side_effect = RuntimeError("Fatal LLM service disconnect")
        result = await worker.execute(brief=sample_brief, shared_budget=budget)

    # Worker must isolate failure and NOT bubble exception
    assert result.status == WorkerStatus.FAILED
    assert "Fatal LLM service disconnect" in (result.error_message or "")
