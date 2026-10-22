"""Unit tests for irreversible gated tools (refunds, deletions, purges)."""

from __future__ import annotations

import pytest

from research_and_support_agent.memory.manager import AgentMemoryManager
from research_and_support_agent.models import CustomerAccount
from research_and_support_agent.tools.irreversible import (
    delete_customer_data,
    delete_episodic_memory_all,
    delete_longterm_memory_all,
    delete_longterm_memory_specific,
    execute_refund,
    purge_expired,
)


@pytest.mark.asyncio
async def test_execute_refund(
    sample_accounts: dict[str, CustomerAccount],
    memory_manager: AgentMemoryManager,
) -> None:
    """Verify execute_refund credits account and logs to episodic memory."""
    res = await execute_refund(
        customer_id="C-101",
        amount_usd=25.0,
        reason="Defective accessory",
        account_store=sample_accounts,
        memory_manager=memory_manager,
    )
    assert res["status"] == "completed"
    assert res["refund_amount_usd"] == 25.0
    assert res["new_balance_usd"] == 125.0


@pytest.mark.asyncio
async def test_delete_customer_data(
    sample_accounts: dict[str, CustomerAccount],
) -> None:
    """Verify delete_customer_data redacts profile fields."""
    res = await delete_customer_data(
        customer_id="C-102",
        data_scope="profile",
        account_store=sample_accounts,
    )
    assert res["status"] == "data_deleted"
    assert sample_accounts["C-102"].name == "[REDACTED]"
    assert sample_accounts["C-102"].email == "redacted_C-102@deleted.local"


@pytest.mark.asyncio
async def test_memory_deletions_irreversible(
    memory_manager: AgentMemoryManager,
) -> None:
    """Verify long-term and episodic memory deletions."""
    # Seed a memory
    saved = await memory_manager.save_longterm_memory("U-IRR", "Temporary secret fact")
    rec_id = saved["record_id"]

    # Delete specific
    del_spec = await delete_longterm_memory_specific("U-IRR", rec_id, memory_manager=memory_manager)
    assert del_spec["status"] == "deleted"

    # Delete all
    del_all = await delete_longterm_memory_all("U-IRR", memory_manager=memory_manager)
    assert del_all["status"] == "all_longterm_deleted"

    # Delete episodic all
    ep_del = await delete_episodic_memory_all("U-IRR", memory_manager=memory_manager)
    assert ep_del["status"] == "all_episodes_deleted"

    # Purge expired
    purge_res = await purge_expired("U-IRR", memory_manager=memory_manager)
    assert purge_res["status"] == "expired_purged"
