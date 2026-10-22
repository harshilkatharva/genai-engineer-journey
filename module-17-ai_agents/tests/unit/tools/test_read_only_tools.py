"""Unit tests for read-only tools."""

from __future__ import annotations

from typing import Any

import pytest

from research_and_support_agent.memory.manager import AgentMemoryManager
from research_and_support_agent.models import CustomerAccount, SupportTicket
from research_and_support_agent.tools.read_only import (
    get_episodic_memory,
    get_longterm_memory,
    lookup_customer_account,
    search_knowledge_base,
    ticket_status,
)


@pytest.mark.asyncio
async def test_search_knowledge_base_formatting(mock_retrieval_service: Any) -> None:
    """Verify search_knowledge_base returns sanitized XML tagged chunks."""
    kb_res = await search_knowledge_base("warranty", retrieval_service=mock_retrieval_service)
    assert "warranty" in kb_res.lower()
    assert "<untrusted_content" in kb_res
    assert "</untrusted_content>" in kb_res


@pytest.mark.asyncio
async def test_lookup_customer_account_found_and_missing(
    sample_accounts: dict[str, CustomerAccount],
) -> None:
    """Verify customer lookup for existing and missing accounts."""
    # Existing account
    cust = await lookup_customer_account("C-101", account_store=sample_accounts)
    assert cust["customer_id"] == "C-101"
    assert cust["tier"] == "Gold"
    assert cust["balance_usd"] == 100.0

    # Missing account
    missing = await lookup_customer_account("C-NONEXISTENT", account_store=sample_accounts)
    assert "error" in missing
    assert "not found" in missing["error"].lower()


@pytest.mark.asyncio
async def test_ticket_status_lookup(
    sample_tickets: dict[str, SupportTicket],
) -> None:
    """Verify ticket status lookup for existing and missing tickets."""
    t_res = await ticket_status("T-101", ticket_store=sample_tickets)
    assert t_res["ticket_id"] == "T-101"
    assert t_res["status"] == "open"
    assert t_res["priority"] == "high"

    missing = await ticket_status("T-MISSING", ticket_store=sample_tickets)
    assert "error" in missing


@pytest.mark.asyncio
async def test_memory_read_tools(
    memory_manager: AgentMemoryManager,
) -> None:
    """Verify get_longterm_memory and get_episodic_memory return lists."""
    lt_res = await get_longterm_memory("U-1", "preferences", memory_manager=memory_manager)
    assert isinstance(lt_res, list)

    ep_res = await get_episodic_memory("U-1", memory_manager=memory_manager)
    assert isinstance(ep_res, list)
