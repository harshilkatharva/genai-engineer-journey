"""Integration tests executing real EvalSample definitions from the benchmark dataset."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from genai_core import (
    LLMManagerResponse,
    ToolCall,
    ToolLoopStatus,
)

from research_and_support_agent.agent import BoundedAgent
from research_and_support_agent.eval.dataset import get_15_benchmark_samples


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_eval_sample_ticket_status_flow(agent: BoundedAgent) -> None:
    """Run real EvalSample goal_01_ticket_status and verify end-to-end execution flow."""
    samples = get_15_benchmark_samples()
    sample = next(s for s in samples if s.id == "goal_01_ticket_status")

    # Step 1: Model calls ticket_status tool
    # Step 2: Model outputs final answer with required keywords
    call = ToolCall(id="c_ticket", name="ticket_status", arguments={"ticket_id": "T-101"})
    resp_step1 = LLMManagerResponse(
        text=None,
        model="gemini-3.5-flash-lite",
        tool_calls=[call],
        usage={"input_tokens": 120, "output_tokens": 30},
    )
    resp_step2 = LLMManagerResponse(
        text="Ticket T-101 is currently OPEN and assigned to tier1_support with high priority.",
        model="gemini-3.5-flash-lite",
        tool_calls=[],
        usage={"input_tokens": 180, "output_tokens": 40},
    )

    agent.llm_service.complete_tool = AsyncMock(side_effect=[resp_step1, resp_step2])

    result = await agent.run(query=sample.query)

    assert result.status == ToolLoopStatus.COMPLETED
    assert result.iterations == 2

    # Verify expected tool was called
    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert len(tool_msgs) >= 1
    assert tool_msgs[0].tool_name == "ticket_status"

    # Verify required keywords from sample.expected_answer
    assert sample.expected_answer and sample.expected_answer.required
    ans_lower = (result.final_text or "").lower()
    for req in sample.expected_answer.required:
        assert req.lower() in ans_lower, f"Missing required term: {req}"

    # Verify transcript recording
    assert result.transcript.total_steps >= 2
    assert result.transcript.status == "completed"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_eval_sample_customer_lookup_flow(agent: BoundedAgent) -> None:
    """Run real EvalSample goal_02_customer_lookup and verify end-to-end execution flow."""
    samples = get_15_benchmark_samples()
    sample = next(s for s in samples if s.id == "goal_02_customer_lookup")

    call = ToolCall(id="c_cust", name="lookup_customer_account", arguments={"customer_id": "C-101"})
    resp_step1 = LLMManagerResponse(
        text=None,
        model="gemini-3.5-flash-lite",
        tool_calls=[call],
        usage={"input_tokens": 110, "output_tokens": 25},
    )
    resp_step2 = LLMManagerResponse(
        text="Customer C-101 is Alice Smith in Gold tier with account balance of $100.00.",
        model="gemini-3.5-flash-lite",
        tool_calls=[],
        usage={"input_tokens": 160, "output_tokens": 35},
    )

    agent.llm_service.complete_tool = AsyncMock(side_effect=[resp_step1, resp_step2])

    result = await agent.run(query=sample.query)

    assert result.status == ToolLoopStatus.COMPLETED
    assert result.iterations == 2

    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert len(tool_msgs) >= 1
    assert tool_msgs[0].tool_name == "lookup_customer_account"

    # Verify required keywords
    assert sample.expected_answer and sample.expected_answer.required
    ans_lower = (result.final_text or "").lower()
    for req in sample.expected_answer.required:
        assert req.lower() in ans_lower, f"Missing required term: {req}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_eval_sample_kb_retrieval_flow(agent: BoundedAgent) -> None:
    """Run real EvalSample goal_03_kb_retrieval and verify end-to-end execution flow."""
    samples = get_15_benchmark_samples()
    sample = next(s for s in samples if s.id == "goal_03_kb_retrieval")

    call = ToolCall(
        id="c_kb",
        name="search_knowledge_base",
        arguments={"query": "hardware warranty return window policy"},
    )
    resp_step1 = LLMManagerResponse(
        text=None,
        model="gemini-3.5-flash-lite",
        tool_calls=[call],
        usage={"input_tokens": 140, "output_tokens": 30},
    )
    resp_step2 = LLMManagerResponse(
        text="Under company policy, all hardware items carry a warranty return window of 30 days.",
        model="gemini-3.5-flash-lite",
        tool_calls=[],
        usage={"input_tokens": 220, "output_tokens": 45},
    )

    agent.llm_service.complete_tool = AsyncMock(side_effect=[resp_step1, resp_step2])

    result = await agent.run(query=sample.query)

    assert result.status == ToolLoopStatus.COMPLETED
    assert result.iterations == 2

    # Verify search_knowledge_base tool was executed
    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert len(tool_msgs) >= 1
    assert tool_msgs[0].tool_name == "search_knowledge_base"

    # Verify required keywords
    assert sample.expected_answer and sample.expected_answer.required
    ans_lower = (result.final_text or "").lower()
    for req in sample.expected_answer.required:
        assert req.lower() in ans_lower, f"Missing required term: {req}"
