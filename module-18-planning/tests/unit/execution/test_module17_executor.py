"""Unit tests for Module17StepExecutor and capability filtering."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from plan_and_execute_research_assistant.core.config import AssistantConfig
from plan_and_execute_research_assistant.execution.module17_executor import Module17StepExecutor
from plan_and_execute_research_assistant.models.enums import StepStatus
from plan_and_execute_research_assistant.models.plan import PlanStep


def test_capabilities_filter_episodic_when_disabled() -> None:
    """Verify episodic tools are filtered out when enable_episodic_similarity_search is False."""
    config = AssistantConfig(enable_episodic_similarity_search=False)
    agent = MagicMock()
    mock_registry = MagicMock()
    mock_tool_1 = MagicMock()
    mock_tool_1.name = "lookup_customer_account"
    mock_tool_1.description = "Lookup customer details"

    mock_tool_2 = MagicMock()
    mock_tool_2.name = "get_episodic_memory"
    mock_tool_2.description = "Retrieve past action history"

    mock_tool_3 = MagicMock()
    mock_tool_3.name = "search_knowledge_base"
    mock_tool_3.description = "Search KB"

    mock_registry.list_tools.return_value = [mock_tool_1, mock_tool_2, mock_tool_3]
    agent.tool_registry = mock_registry

    executor = Module17StepExecutor(agent=agent, config=config)
    caps = executor.get_available_capabilities()

    assert "lookup_customer_account" in caps
    assert "search_knowledge_base" in caps
    assert "get_episodic_memory" not in caps


def test_capabilities_preserve_episodic_when_enabled() -> None:
    """Verify episodic tools are preserved when enable_episodic_similarity_search is True."""
    config = AssistantConfig(enable_episodic_similarity_search=True)
    agent = MagicMock()
    mock_registry = MagicMock()
    mock_tool = MagicMock()
    mock_tool.name = "get_episodic_memory"
    mock_tool.description = "Retrieve past action history"

    mock_registry.list_tools.return_value = [mock_tool]
    agent.tool_registry = mock_registry

    executor = Module17StepExecutor(agent=agent, config=config)
    caps = executor.get_available_capabilities()

    assert "get_episodic_memory" in caps


def test_format_step_prompt_includes_user_context() -> None:
    """Verify format_step_prompt formats objective, hints, and entity context."""
    executor = Module17StepExecutor(agent=MagicMock())
    step = PlanStep(
        id="step_1",
        description="Lookup customer details for C-104",
        tool_name_hint="lookup_customer_account",
        success_criteria="Customer details retrieved",
    )
    prompt = executor.format_step_prompt(
        step=step,
        predecessor_context={"step_0": "Verified account exists"},
        user_id="C-104",
        retry_feedback="Previous attempt returned empty",
    )

    assert "STEP IDENTIFIER: step_1" in prompt
    assert "OBJECTIVE: Lookup customer details for C-104" in prompt
    assert "TARGET USER / ENTITY CONTEXT: C-104" in prompt
    assert "RECOMMENDED TOOL HINT: lookup_customer_account" in prompt
    assert "PREDECESSOR CONTEXT (Completed Dependency Outputs):" in prompt
    assert "- [step_0]: Verified account exists" in prompt
    assert "CRITICAL VERIFICATION FEEDBACK FROM PREVIOUS ATTEMPT:" in prompt
    assert "Previous attempt returned empty" in prompt


@pytest.mark.asyncio
async def test_execute_step_success() -> None:
    """Verify successful step execution returning StepExecutionOutput."""
    agent = MagicMock()
    mock_result = MagicMock()
    mock_result.status = "completed"
    mock_result.final_text = "Account C-104 has balance $0.0 and active risk flag."
    mock_result.total_tokens = 250
    mock_result.total_cost_usd = 0.0001
    mock_result.error_message = None
    agent.run = AsyncMock(return_value=mock_result)

    executor = Module17StepExecutor(agent=agent)
    step = PlanStep(
        id="step_1",
        description="Lookup customer account",
        success_criteria="Account details retrieved",
    )
    output = await executor.execute_step(step=step, predecessor_context={}, user_id="C-104")

    assert output.step_id == "step_1"
    assert output.status == StepStatus.COMPLETED
    assert output.output_text == "Account C-104 has balance $0.0 and active risk flag."
    assert output.tokens_used == 250
    assert output.cost_usd == 0.0001


@pytest.mark.asyncio
async def test_execute_step_truncates_web_search_output() -> None:
    """Verify web-retrieved output exceeding max_web_retrieved_tokens is truncated with tiktoken."""
    from plan_and_execute_research_assistant.core.tokens import count_tokens

    config = AssistantConfig(max_web_retrieved_tokens=100)
    agent = MagicMock()
    mock_result = MagicMock()
    mock_result.status = "completed"
    # Create output with ~300 tokens
    mock_result.final_text = "web document word " * 150
    mock_result.total_tokens = 500
    mock_result.total_cost_usd = 0.001
    mock_result.error_message = None
    agent.run = AsyncMock(return_value=mock_result)

    executor = Module17StepExecutor(agent=agent, config=config)
    step = PlanStep(
        id="step_web",
        description="Search web for pricing",
        tool_name_hint="search_web",
        success_criteria="Pricing retrieved",
    )
    output = await executor.execute_step(step=step, predecessor_context={}, user_id="user_test")

    assert output.status == StepStatus.COMPLETED
    assert "truncated to 100 tokens limit" in output.output_text
    prefix = output.output_text.split("\n... [Web-retrieved response truncated")[0]
    assert count_tokens(prefix) <= 100


@pytest.mark.asyncio
async def test_execute_step_approval_required_is_success() -> None:
    """Verify approval_required status is treated as successful hand-off deliverable."""
    agent = MagicMock()
    mock_result = MagicMock()
    mock_result.status = "approval_required"
    mock_result.final_text = None
    mock_req = MagicMock()
    mock_req.tool_name = "execute_refund"
    mock_req.request_id = "req_12345"
    mock_req.arguments = {"customer_id": "C-102", "amount_usd": 50.0}
    mock_req.reason = "Refund exceeds automated threshold"
    mock_req.risk_level.value = "irreversible"
    mock_result.pending_approval = mock_req
    mock_result.total_tokens = 400
    mock_result.total_cost_usd = 0.0002
    mock_result.error_message = None
    agent.run = AsyncMock(return_value=mock_result)

    executor = Module17StepExecutor(agent=agent)
    step = PlanStep(
        id="step_refund",
        description="Execute support refund of $50.00 USD for customer C-102",
        tool_name_hint="execute_refund",
        success_criteria="Refund successfully executed for customer C-102",
    )
    output = await executor.execute_step(step=step, predecessor_context={}, user_id="C-102")

    assert output.status == StepStatus.COMPLETED
    assert "execute_refund" in output.output_text
    assert "req_12345" in output.output_text
    assert "paused for human authorization" in output.output_text


def test_format_step_prompt_truncates_large_predecessor_context() -> None:
    """Verify format_step_prompt truncates large predecessor summaries to 1,000 tokens per dep (PLN-011)."""
    agent = MagicMock()
    executor = Module17StepExecutor(agent=agent)
    step = PlanStep(
        id="step_2",
        description="Synthesize results",
        success_criteria="Synthesis complete",
    )
    huge_summary = "word " * 2500  # ~2500 tokens
    prompt = executor.format_step_prompt(
        step=step,
        predecessor_context={"step_1": huge_summary},
        user_id="user_test",
    )
    assert "[step_1]:" in prompt
    assert "truncated to 1,000 tokens limit" in prompt
