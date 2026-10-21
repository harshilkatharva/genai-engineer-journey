"""Unit tests for the 16 operational domain tools in Module 17."""

from __future__ import annotations

from typing import Any

import pytest
from genai_core import RiskLevel

from research_and_support_agent.memory.manager import AgentMemoryManager
from research_and_support_agent.models import (
    CustomerAccount,
    ReversalAuditRecord,
    SupportTicket,
)
from research_and_support_agent.tools.irreversible import (
    delete_customer_data,
    delete_longterm_memory_all,
    delete_longterm_memory_specific,
    execute_refund,
    purge_expired,
)
from research_and_support_agent.tools.read_only import (
    get_episodic_memory,
    get_longterm_memory,
    lookup_customer_account,
    search_knowledge_base,
    ticket_status,
)
from research_and_support_agent.tools.registry import create_agent_tool_registry
from research_and_support_agent.tools.reversible import (
    escalate_to_tier2,
    save_longterm_memory,
    update_ticket_status,
)


@pytest.mark.asyncio
async def test_tool_registry_registration(memory_manager: AgentMemoryManager) -> None:
    """Verify that all 16 tools are registered with appropriate risk classifications."""
    registry = create_agent_tool_registry(memory_manager=memory_manager)
    specs = registry.get_specs()
    assert len(specs) == 16

    # Verify risk classifications
    read_only_tools = registry.get_specs(max_risk=RiskLevel.READ_ONLY)
    assert len(read_only_tools) == 6

    reversible_tools = registry.get_specs(max_risk=RiskLevel.REVERSIBLE_WRITE)
    assert len(reversible_tools) == 9  # 6 read-only + 3 reversible


@pytest.mark.asyncio
async def test_read_only_tools(
    mock_retrieval_service: Any,
    memory_manager: AgentMemoryManager,
    sample_accounts: dict[str, CustomerAccount],
    sample_tickets: dict[str, SupportTicket],
) -> None:
    """Test all read-only operational tools."""
    # 1. search_knowledge_base
    kb_res = await search_knowledge_base("warranty", retrieval_service=mock_retrieval_service)
    assert "warranty" in kb_res.lower()
    assert "<untrusted_content" in kb_res

    # 2. lookup_customer_account
    cust = await lookup_customer_account("C-101", account_store=sample_accounts)
    assert cust["customer_id"] == "C-101"
    assert cust["tier"] == "Gold"

    missing = await lookup_customer_account("C-NONEXISTENT", account_store=sample_accounts)
    assert "error" in missing

    # 3. ticket_status
    t_res = await ticket_status("T-101", ticket_store=sample_tickets)
    assert t_res["ticket_id"] == "T-101"
    assert t_res["status"] == "open"

    # 4. get_longterm_memory & get_episodic_memory (empty initially)
    lt_res = await get_longterm_memory("U-1", "pref", memory_manager=memory_manager)
    assert isinstance(lt_res, list)

    ep_res = await get_episodic_memory("U-1", memory_manager=memory_manager)
    assert isinstance(ep_res, list)


@pytest.mark.asyncio
async def test_reversible_tools(
    memory_manager: AgentMemoryManager,
    sample_tickets: dict[str, SupportTicket],
    audit_log: list[ReversalAuditRecord],
) -> None:
    """Test reversible operational tools and undo tracking."""
    # 1. save_longterm_memory
    saved = await save_longterm_memory(
        "U-100", "Customer speaks German", category="preference", memory_manager=memory_manager
    )
    assert saved["status"] == "saved"
    assert saved["user_id"] == "U-100"

    # 2. update_ticket_status
    upd = await update_ticket_status(
        "T-101",
        "in_progress",
        comment="Assigned to agent",
        ticket_store=sample_tickets,
        audit_log=audit_log,
    )
    assert upd["previous_status"] == "open"
    assert upd["current_status"] == "in_progress"
    assert upd["reversible"] is True
    assert len(audit_log) == 1

    # 3. escalate_to_tier2
    esc = await escalate_to_tier2(
        "T-101", "Kernel panic", urgency="high", ticket_store=sample_tickets, audit_log=audit_log
    )
    assert esc["status"] == "escalated_tier2"
    assert esc["assigned_team"] == "tier2_engineering"
    assert esc["reversible"] is True
    assert len(audit_log) == 2


@pytest.mark.asyncio
async def test_irreversible_tools(
    memory_manager: AgentMemoryManager,
    sample_accounts: dict[str, CustomerAccount],
) -> None:
    """Test irreversible tools: refunds, customer data delete, and memory deletions."""
    # 1. execute_refund
    ref = await execute_refund(
        customer_id="C-101",
        amount_usd=25.0,
        reason="Shipping delay",
        account_store=sample_accounts,
        memory_manager=memory_manager,
    )
    assert ref["status"] == "completed"
    assert ref["new_balance_usd"] == 125.0

    # 2. delete_customer_data
    del_res = await delete_customer_data("C-102", "profile", account_store=sample_accounts)
    assert del_res["status"] == "data_deleted"
    assert sample_accounts["C-102"].name == "[REDACTED]"

    # 3. Memory deletions
    # Seed a memory first
    await memory_manager.save_longterm_memory("U-DEL", "Some secret fact")
    del_all = await delete_longterm_memory_all("U-DEL", memory_manager=memory_manager)
    assert del_all["status"] == "all_longterm_deleted"

    # Specific memory delete
    saved2 = await memory_manager.save_longterm_memory("U-DEL2", "Temporary fact")
    rec_id = saved2["record_id"]
    del_spec = await delete_longterm_memory_specific(
        "U-DEL2", rec_id, memory_manager=memory_manager
    )
    assert del_spec["status"] == "deleted"

    # Purge expired
    purge_res = await purge_expired("U-DEL", memory_manager=memory_manager)
    assert purge_res["status"] == "expired_purged"
