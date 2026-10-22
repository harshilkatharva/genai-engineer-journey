"""Integration tests verifying bounded execution guardrails."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest
from genai_core import (
    LLMManagerResponse,
    ToolBudget,
    ToolCall,
    ToolLoopStatus,
)

from research_and_support_agent.agent import BoundedAgent


@pytest.mark.integration
@pytest.mark.asyncio
async def test_step_budget_exhaustion(agent: BoundedAgent) -> None:
    """Verify agent terminates cleanly with MAX_ITERATIONS_REACHED when step cap is hit."""
    # Mock LLM to continuously request a tool call
    call = ToolCall(id="call_1", name="ticket_status", arguments={"ticket_id": "T-101"})
    resp = LLMManagerResponse(
        text=None,
        model="gemini-1.5-flash",
        tool_calls=[call],
        usage={"input_tokens": 20, "output_tokens": 10},
    )
    agent.llm_service.complete_tool = AsyncMock(return_value=resp)

    # Set strict step limit = 2
    budget = ToolBudget(max_steps=2)
    result = await agent.run(query="Audit ticket T-101", budget=budget)

    assert result.status == ToolLoopStatus.MAX_ITERATIONS_REACHED
    assert result.iterations == 2
    assert budget.is_step_budget_exhausted() is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_wall_clock_deadline_exhaustion(agent: BoundedAgent) -> None:
    """Verify agent halts cleanly when the wall-clock deadline passes."""
    call = ToolCall(id="call_1", name="ticket_status", arguments={"ticket_id": "T-101"})
    resp = LLMManagerResponse(
        text=None,
        model="gemini-1.5-flash",
        tool_calls=[call],
        usage={"input_tokens": 20, "output_tokens": 10},
    )

    async def slow_complete(*args, **kwargs):
        await asyncio.sleep(0.15)  # Simulate delay exceeding deadline
        return resp

    agent.llm_service.complete_tool = AsyncMock(side_effect=slow_complete)

    # Set tight deadline of 0.1s
    budget = ToolBudget(max_steps=5, deadline_monotonic=time.monotonic() + 0.1)
    result = await agent.run(query="Audit ticket T-101", budget=budget)

    assert result.status == ToolLoopStatus.DEADLINE_EXCEEDED
    assert budget.is_deadline_exceeded() is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_token_budget_exhaustion(agent: BoundedAgent) -> None:
    """Verify agent halts cleanly when cumulative token usage exceeds limit."""
    call = ToolCall(id="call_1", name="ticket_status", arguments={"ticket_id": "T-101"})
    # First response uses 120 tokens, exceeding 100 token budget
    resp = LLMManagerResponse(
        text=None,
        model="gemini-1.5-flash",
        tool_calls=[call],
        usage={"input_tokens": 80, "output_tokens": 40},
    )
    agent.llm_service.complete_tool = AsyncMock(return_value=resp)

    budget = ToolBudget(max_steps=5, max_tokens=100)
    result = await agent.run(query="Audit ticket T-101", budget=budget)

    assert result.status == ToolLoopStatus.BUDGET_EXCEEDED
    assert budget.current_tokens >= 100


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cost_budget_exhaustion(agent: BoundedAgent) -> None:
    """Verify agent halts cleanly when cumulative cost exceeds limit."""
    call = ToolCall(id="call_1", name="ticket_status", arguments={"ticket_id": "T-101"})
    resp = LLMManagerResponse(
        text=None,
        model="gemini-1.5-flash",
        tool_calls=[call],
        usage={"input_tokens": 500, "output_tokens": 200},
    )
    agent.llm_service.complete_tool = AsyncMock(return_value=resp)

    # Set microscopic cost budget of $0.000001
    budget = ToolBudget(max_steps=5, max_cost_usd=0.000001)
    budget.record_step(tokens=700, cost_usd=0.00005)  # Pre-exhaust cost
    result = await agent.run(query="Audit ticket T-101", budget=budget)

    assert result.status == ToolLoopStatus.BUDGET_EXCEEDED
    assert budget.is_cost_budget_exhausted() is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_repeated_loop_detection(agent: BoundedAgent) -> None:
    """Verify LoopDetector catches 3 identical consecutive tool calls and injects guidance."""
    # LLM issues identical tool call repeatedly
    call = ToolCall(id="call_1", name="ticket_status", arguments={"ticket_id": "T-101"})
    resp = LLMManagerResponse(
        text=None,
        model="gemini-1.5-flash",
        tool_calls=[call],
        usage={"input_tokens": 20, "output_tokens": 10},
    )
    agent.llm_service.complete_tool = AsyncMock(return_value=resp)

    budget = ToolBudget(max_steps=4)
    result = await agent.run(query="Audit ticket", budget=budget)

    # Check that loop detection message was injected
    tool_messages = [m for m in result.messages if m.role == "tool"]
    loop_alerts = [m for m in tool_messages if "Error: Loop detected" in (m.content or "")]
    assert len(loop_alerts) >= 1
    assert "Repeated identical call" in loop_alerts[0].content
