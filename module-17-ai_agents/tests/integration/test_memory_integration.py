"""Integration tests for persistent memory, pre-turn recall, and episodic idempotency."""

from __future__ import annotations

import pytest
from genai_core import (
    ToolLoopStatus,
)

from research_and_support_agent.agent import BoundedAgent
from research_and_support_agent.memory.manager import AgentMemoryManager


@pytest.mark.asyncio
async def test_preturn_memory_recall_context_injection(
    agent: BoundedAgent,
    memory_manager: AgentMemoryManager,
) -> None:
    """Verify that user facts stored in long-term memory are recalled and injected into system prompt."""
    user_id = "U-PREF-99"
    # Store a long-term preference
    await memory_manager.save_longterm_memory(
        user_id=user_id,
        content="Customer requires German invoices and prefers email summaries.",
        category="preference",
    )

    # Run agent
    res = await agent.run(query="How should we send invoices?", user_id=user_id)
    assert res.status == ToolLoopStatus.COMPLETED

    # Verify the system message contained the recalled memory
    system_messages = [m for m in res.messages if m.role == "system"]
    assert len(system_messages) >= 1
    sys_content = system_messages[0].content
    assert "German invoices" in sys_content


@pytest.mark.asyncio
async def test_episodic_action_logging_and_idempotency_block(
    agent: BoundedAgent,
) -> None:
    """Verify that episodic memory logs tool execution and blocks duplicate sensitive actions."""
    customer_id = "C-101"
    refund_args = {"customer_id": customer_id, "amount_usd": 30.0, "reason": "damaged item"}

    # Execute first refund directly through tool implementation
    tool_ref = agent.tool_registry.get("execute_refund")
    assert tool_ref is not None

    res1 = await tool_ref.run(**refund_args)
    assert res1["status"] == "completed"
    assert res1["refund_amount_usd"] == 30.0

    # Attempt second refund with identical parameters within same retention window
    res2 = await tool_ref.run(**refund_args)
    assert res2.get("status") == "duplicate_blocked"
    assert "Duplicate action detected" in res2["error"]


@pytest.mark.asyncio
async def test_memory_deletion_tools(
    agent: BoundedAgent,
    memory_manager: AgentMemoryManager,
) -> None:
    """Verify selective and bulk memory deletion tools."""
    user_id = "U-CLEANUP"

    # Save multiple long-term memories
    m1 = await memory_manager.save_longterm_memory(user_id, "Fact A")
    await memory_manager.save_longterm_memory(user_id, "Fact B")

    tool_del_spec = agent.tool_registry.get("delete_longterm_memory_specific")
    assert tool_del_spec is not None
    del_res = await tool_del_spec.run(user_id=user_id, record_id=m1["record_id"])
    assert del_res["status"] == "deleted"

    # Verify Fact B remains
    remaining = await memory_manager.get_longterm_memory(user_id, "Fact")
    assert len(remaining) == 1

    # Bulk delete
    tool_del_all = agent.tool_registry.get("delete_longterm_memory_all")
    assert tool_del_all is not None
    del_all_res = await tool_del_all.run(user_id=user_id)
    assert del_all_res["status"] == "all_longterm_deleted"

    # Verify no records remain
    assert len(await memory_manager.get_longterm_memory(user_id, "Fact")) == 0
