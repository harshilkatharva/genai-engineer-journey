from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

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


class DataFetchArgs(BaseModel):
    dataset_name: str


class CalculationArgs(BaseModel):
    numbers: list[float]


class ArchiveArgs(BaseModel):
    archive_id: str


@pytest.mark.asyncio
async def test_end_to_end_tool_workflow_with_approval_and_budget():
    # 1. Setup Registry with layered risk tools
    registry = ToolRegistry()

    @tool(args_schema=DataFetchArgs, risk_level=RiskLevel.READ_ONLY, tags=["data"])
    def fetch_data(dataset_name: str) -> list[float]:
        return [10.5, 20.0, 30.5, 40.0]

    @tool(
        args_schema=CalculationArgs,
        risk_level=RiskLevel.READ_ONLY,
        tags=["compute"],
    )
    def compute_average(numbers: list[float]) -> float:
        return sum(numbers) / len(numbers) if numbers else 0.0

    @tool(
        args_schema=ArchiveArgs,
        risk_level=RiskLevel.IRREVERSIBLE,
        tags=["admin"],
    )
    async def archive_dataset(archive_id: str) -> dict[str, str]:
        return {"status": "archived", "id": archive_id}

    registry.register(fetch_data)
    registry.register(compute_average)
    registry.register(archive_dataset)

    # 2. Setup Approval Store and Executor
    approval_store = InMemoryApprovalStore()
    executor = ToolExecutor(
        registry=registry,
        approval_store=approval_store,
        require_approval_for={RiskLevel.IRREVERSIBLE},
        max_output_chars=5000,
        default_timeout_s=5.0,
    )

    # 3. Setup Mock LLM simulating multi-step agent flow
    # Turn 1: Call fetch_data
    resp1 = LLMManagerResponse(
        text=None,
        tool_calls=[
            ToolCall(
                id="call_fetch",
                name="fetch_data",
                arguments={"dataset_name": "sales_q3"},
            )
        ],
        usage={"input_tokens": 50, "output_tokens": 20},
    )
    # Turn 2: Call compute_average
    resp2 = LLMManagerResponse(
        text=None,
        tool_calls=[
            ToolCall(
                id="call_calc",
                name="compute_average",
                arguments={"numbers": [10.5, 20.0, 30.5, 40.0]},
            )
        ],
        usage={"input_tokens": 70, "output_tokens": 25},
    )
    # Turn 3: Call archive_dataset (IRREVERSIBLE - pauses for approval!)
    resp3 = LLMManagerResponse(
        text=None,
        tool_calls=[
            ToolCall(
                id="call_archive",
                name="archive_dataset",
                arguments={"archive_id": "sales_q3"},
            )
        ],
        usage={"input_tokens": 90, "output_tokens": 20},
    )
    # Turn 4 (after resume): Provide final summary
    resp4 = LLMManagerResponse(
        text="The average sales figure is 25.25. Dataset sales_q3 has been archived successfully.",
        tool_calls=[],
        usage={"input_tokens": 120, "output_tokens": 30},
    )

    mock_llm = AsyncMock(spec=LLMService)
    mock_llm.complete_tool.side_effect = [resp1, resp2, resp3, resp4]

    budget = ToolBudget(max_steps=10, max_tokens=2000)
    runner = ToolLoopRunner(
        llm_service=mock_llm,
        executor=executor,
    )

    initial_messages = [
        ChatMessage(
            role="user",
            content="Fetch sales_q3, compute average, and archive the dataset.",
        )
    ]

    # Run the loop: executes turns 1 and 2, then pauses on turn 3
    paused_result = await runner.run(messages=initial_messages, budget=budget)

    assert paused_result.status == ToolLoopStatus.APPROVAL_REQUIRED
    assert paused_result.pending_approval is not None
    assert paused_result.pending_approval.tool_name == "archive_dataset"
    assert paused_result.iterations == 3

    # Check that approval store tracked the pending request
    pending_list = await approval_store.list_pending_requests()
    assert len(pending_list) == 1
    assert pending_list[0].request_id == paused_result.pending_approval.request_id

    # Resume the loop with human approval
    decision = ApprovalDecision(approved=True, reason="Archival authorized")
    resumed_result = await runner.resume(
        decision=decision,
        request=paused_result.pending_approval,
        messages=paused_result.messages,
        budget=budget,
        initial_iterations=paused_result.iterations,
        initial_tool_calls=paused_result.total_tool_calls,
    )

    assert resumed_result.status == ToolLoopStatus.COMPLETED
    assert (
        resumed_result.final_text
        == "The average sales figure is 25.25. Dataset sales_q3 has been archived successfully."
    )
    assert resumed_result.total_tool_calls >= 3

    # Verify all tool results are present in message history
    tool_messages = [m for m in resumed_result.messages if m.role == "tool"]
    assert len(tool_messages) == 3
    assert tool_messages[0].tool_name == "fetch_data"
    assert tool_messages[1].tool_name == "compute_average"
    assert tool_messages[2].tool_name == "archive_dataset"

    # Verify decision stored
    stored_decision = await approval_store.get_decision(paused_result.pending_approval.request_id)
    assert stored_decision is not None
    assert stored_decision.approved is True
