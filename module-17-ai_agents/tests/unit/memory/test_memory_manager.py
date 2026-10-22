"""Unit tests for AgentMemoryManager pre-turn recall, episodic checks, and deletions."""

from __future__ import annotations

import pytest

from research_and_support_agent.memory.manager import AgentMemoryManager


@pytest.mark.asyncio
async def test_memory_manager_empty_user_context(memory_manager: AgentMemoryManager) -> None:
    """Verify empty user_id returns empty prompt context string."""
    ctx = await memory_manager.get_context_for_prompt(user_id="", goal="help me")
    assert ctx == ""


@pytest.mark.asyncio
async def test_memory_manager_recall_formatting(memory_manager: AgentMemoryManager) -> None:
    """Verify stored user memories are retrieved and formatted cleanly."""
    await memory_manager.save_longterm_memory(
        user_id="U-TEST-1",
        content="Customer requested communications in French",
        category="preference",
    )

    ctx = await memory_manager.get_context_for_prompt(
        user_id="U-TEST-1",
        goal="language preference",
    )
    assert "[Recalled Long-Term User Facts & Preferences]:" in ctx
    assert "Customer requested communications in French" in ctx


@pytest.mark.asyncio
async def test_memory_manager_episodic_action_check(memory_manager: AgentMemoryManager) -> None:
    """Verify episodic action logging and duplicate detection."""
    # First action check: should be allowed ("execute")
    check1 = await memory_manager.check_action_idempotency(
        user_id="U-TEST-2",
        action_type="execute_refund",
        parameters={"customer_id": "C-1", "amount": 25.0},
    )
    assert check1.status == "execute"

    # Log action completion
    await memory_manager.record_action_outcome(
        user_id="U-TEST-2",
        action_type="execute_refund",
        parameters={"customer_id": "C-1", "amount": 25.0},
        status="success",
    )

    # Second action check with identical parameters: should be blocked ("do_not_repeat")
    check2 = await memory_manager.check_action_idempotency(
        user_id="U-TEST-2",
        action_type="execute_refund",
        parameters={"customer_id": "C-1", "amount": 25.0},
    )
    assert check2.status == "do_not_repeat"


@pytest.mark.asyncio
async def test_memory_manager_delete_operations(memory_manager: AgentMemoryManager) -> None:
    """Verify delete_longterm_all and delete_episodic_all."""
    await memory_manager.save_longterm_memory(
        user_id="U-DEL",
        content="Fact to delete",
    )

    del_res = await memory_manager.delete_longterm_all(user_id="U-DEL")
    assert del_res["user_id"] == "U-DEL"
    assert del_res["status"] == "all_longterm_deleted"

    # Episodic delete
    ep_res = await memory_manager.delete_episodic_all(user_id="U-DEL")
    assert ep_res["user_id"] == "U-DEL"
    assert ep_res["status"] == "all_episodes_deleted"
