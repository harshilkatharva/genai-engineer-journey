from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from genai_core.models import (
    ChatMessage,
    LLMManagerResponse,
    ToolCall,
)
from genai_core.services.llm_services import LLMService
from genai_core.tools import (
    ApprovalDecision,
    InMemoryApprovalStore,
    RiskLevel,
    ToolBudget,
    ToolExecutor,
    ToolLoopRunner,
    ToolLoopStatus,
    ToolRegistry,
    tool,
)


@pytest.fixture
def sample_executor():
    registry = ToolRegistry()

    @tool(name="add")
    def add(a: int, b: int) -> int:
        return a + b

    @tool(name="drop_db", risk_level=RiskLevel.IRREVERSIBLE)
    def drop_db(db_name: str) -> str:
        return f"Database {db_name} dropped"

    registry.register(add)
    registry.register(drop_db)

    store = InMemoryApprovalStore()
    return ToolExecutor(
        registry=registry,
        require_approval_for={RiskLevel.IRREVERSIBLE},
        approval_store=store,
    )


@pytest.mark.asyncio
async def test_tool_loop_immediate_completion(sample_executor: ToolExecutor):
    mock_llm = AsyncMock(spec=LLMService)
    mock_llm.complete_tool.return_value = LLMManagerResponse(
        text="Hello! How can I help you?",
        tool_calls=[],
        model="mock-model",
        usage={"input_tokens": 10, "output_tokens": 5},
    )

    runner = ToolLoopRunner(llm_service=mock_llm, executor=sample_executor)
    messages = [ChatMessage(role="user", content="Hi")]

    result = await runner.run(messages=messages)

    assert result.status == ToolLoopStatus.COMPLETED
    assert result.final_text == "Hello! How can I help you?"
    assert result.iterations == 1
    assert result.total_tool_calls == 0
    assert len(result.messages) == 2
    assert result.messages[-1].content == "Hello! How can I help you?"


@pytest.mark.asyncio
async def test_tool_loop_multi_turn_execution(sample_executor: ToolExecutor):
    mock_llm = AsyncMock(spec=LLMService)

    # First turn: call tool 'add'
    turn1_resp = LLMManagerResponse(
        text=None,
        tool_calls=[ToolCall(id="call_1", name="add", arguments={"a": 10, "b": 20})],
        model="mock-model",
        usage={"input_tokens": 15, "output_tokens": 10},
    )
    # Second turn: provide final answer
    turn2_resp = LLMManagerResponse(
        text="The answer is 30.",
        tool_calls=[],
        model="mock-model",
        usage={"input_tokens": 25, "output_tokens": 8},
    )
    mock_llm.complete_tool.side_effect = [turn1_resp, turn2_resp]

    runner = ToolLoopRunner(llm_service=mock_llm, executor=sample_executor)
    messages = [ChatMessage(role="user", content="What is 10 + 20?")]

    result = await runner.run(messages=messages)

    assert result.status == ToolLoopStatus.COMPLETED
    assert result.final_text == "The answer is 30."
    assert result.iterations == 2
    assert result.total_tool_calls == 1

    # Check conversation history
    # [0] user: "What is 10 + 20?"
    # [1] assistant: "" (or tool calls description)
    # [2] tool: "30"
    # [3] assistant: "The answer is 30."
    assert len(result.messages) == 4
    assert result.messages[2].role == "tool"
    assert result.messages[2].content == "30"
    assert result.messages[3].role == "assistant"


@pytest.mark.asyncio
async def test_tool_loop_max_iterations_reached(sample_executor: ToolExecutor):
    mock_llm = AsyncMock(spec=LLMService)
    # Never returns final answer; always asks for tool
    mock_llm.complete_tool.return_value = LLMManagerResponse(
        text=None,
        tool_calls=[ToolCall(id="call_x", name="add", arguments={"a": 1, "b": 1})],
        model="mock-model",
    )

    runner = ToolLoopRunner(
        llm_service=mock_llm,
        executor=sample_executor,
        max_iterations=3,
        max_repeated_calls=10,  # avoid triggering loop detector before max iterations
    )
    messages = [ChatMessage(role="user", content="Loop forever")]

    result = await runner.run(messages=messages)

    assert result.status == ToolLoopStatus.MAX_ITERATIONS_REACHED
    assert result.iterations == 3


@pytest.mark.asyncio
async def test_tool_loop_deadline_exceeded(sample_executor: ToolExecutor):
    mock_llm = AsyncMock(spec=LLMService)
    runner = ToolLoopRunner(llm_service=mock_llm, executor=sample_executor)

    # Budget with deadline in the past
    budget = ToolBudget(max_steps=5, deadline_monotonic=time.monotonic() - 1.0)
    messages = [ChatMessage(role="user", content="Timeout test")]

    result = await runner.run(messages=messages, budget=budget)

    assert result.status == ToolLoopStatus.DEADLINE_EXCEEDED
    assert mock_llm.complete_tool.call_count == 0


@pytest.mark.asyncio
async def test_tool_loop_token_budget_exceeded(sample_executor: ToolExecutor):
    mock_llm = AsyncMock(spec=LLMService)
    mock_llm.complete_tool.return_value = LLMManagerResponse(
        text=None,
        tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1, "b": 1})],
        usage={"input_tokens": 100, "output_tokens": 100},
    )

    runner = ToolLoopRunner(llm_service=mock_llm, executor=sample_executor)
    budget = ToolBudget(max_steps=10, max_tokens=150)
    messages = [ChatMessage(role="user", content="Token budget test")]

    result = await runner.run(messages=messages, budget=budget)

    assert result.status == ToolLoopStatus.BUDGET_EXCEEDED
    assert result.iterations == 1


@pytest.mark.asyncio
async def test_tool_loop_detection_feedback(sample_executor: ToolExecutor):
    mock_llm = AsyncMock(spec=LLMService)

    # First turn: call add(1, 1)
    call1 = LLMManagerResponse(
        tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1, "b": 1})]
    )
    # Second turn: identical call add(1, 1) -> triggers loop detector (with max_repeated_calls=2)
    call2 = LLMManagerResponse(
        tool_calls=[ToolCall(id="c2", name="add", arguments={"a": 1, "b": 1})]
    )
    # Third turn: model corrects course and finishes
    final_resp = LLMManagerResponse(text="Loop broken, final answer.", tool_calls=[])

    mock_llm.complete_tool.side_effect = [call1, call2, final_resp]

    runner = ToolLoopRunner(
        llm_service=mock_llm,
        executor=sample_executor,
        max_repeated_calls=2,
    )
    messages = [ChatMessage(role="user", content="Detect loop")]

    result = await runner.run(messages=messages)

    assert result.status == ToolLoopStatus.COMPLETED
    assert result.final_text == "Loop broken, final answer."
    # Verify that a loop error message was inserted
    tool_messages = [m for m in result.messages if m.role == "tool"]
    assert any("Loop detected" in m.content for m in tool_messages)


@pytest.mark.asyncio
async def test_tool_loop_approval_pause_and_resume(sample_executor: ToolExecutor):
    mock_llm = AsyncMock(spec=LLMService)

    # First turn: request drop_db (IRREVERSIBLE)
    turn1 = LLMManagerResponse(
        tool_calls=[ToolCall(id="call_drop", name="drop_db", arguments={"db_name": "production"})]
    )
    # Second turn (after resume): model receives drop output and summarizes
    turn2 = LLMManagerResponse(
        text="Successfully dropped database production.",
        tool_calls=[],
    )
    mock_llm.complete_tool.side_effect = [turn1, turn2]

    runner = ToolLoopRunner(llm_service=mock_llm, executor=sample_executor)
    messages = [ChatMessage(role="user", content="Drop the production database")]

    # 1. Run until pause
    pause_result = await runner.run(messages=messages)

    assert pause_result.status == ToolLoopStatus.APPROVAL_REQUIRED
    assert pause_result.pending_approval is not None
    assert pause_result.pending_approval.tool_name == "drop_db"
    assert pause_result.pending_approval.arguments == {"db_name": "production"}

    # 2. Operator resumes with approved decision
    decision = ApprovalDecision(approved=True, reason="Admin authorized via UI")
    resume_result = await runner.resume(
        decision=decision,
        request=pause_result.pending_approval,
        messages=pause_result.messages,
    )

    assert resume_result.status == ToolLoopStatus.COMPLETED
    assert resume_result.final_text == "Successfully dropped database production."
    # Check that tool result was appended
    tool_msgs = [m for m in resume_result.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert "Database production dropped" in tool_msgs[0].content


@pytest.mark.asyncio
async def test_tool_loop_approval_rejection_resume(sample_executor: ToolExecutor):
    mock_llm = AsyncMock(spec=LLMService)

    turn1 = LLMManagerResponse(
        tool_calls=[ToolCall(id="call_drop", name="drop_db", arguments={"db_name": "prod"})]
    )
    turn2 = LLMManagerResponse(
        text="I understand the operation was denied. Database was not dropped.",
        tool_calls=[],
    )
    mock_llm.complete_tool.side_effect = [turn1, turn2]

    runner = ToolLoopRunner(llm_service=mock_llm, executor=sample_executor)
    messages = [ChatMessage(role="user", content="Drop prod db")]

    pause_result = await runner.run(messages=messages)
    assert pause_result.status == ToolLoopStatus.APPROVAL_REQUIRED

    # Operator rejects the operation
    decision = ApprovalDecision(approved=False, reason="Production drops are prohibited")
    resume_result = await runner.resume(
        decision=decision,
        request=pause_result.pending_approval,  # type: ignore[arg-type]
        messages=pause_result.messages,
    )

    assert resume_result.status == ToolLoopStatus.COMPLETED
    assert "operation was denied" in (resume_result.final_text or "")
    # Check that rejection error message was sent as tool result
    tool_msgs = [m for m in resume_result.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert "Production drops are prohibited" in tool_msgs[0].content
