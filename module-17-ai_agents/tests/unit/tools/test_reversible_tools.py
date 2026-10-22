"""Unit tests for reversible write tools and reversal audit tracking."""

from __future__ import annotations

import pytest

from research_and_support_agent.memory.manager import AgentMemoryManager
from research_and_support_agent.models import ReversalAuditRecord, SupportTicket
from research_and_support_agent.tools.reversible import (
    escalate_to_tier2,
    save_longterm_memory,
    update_ticket_status,
)


@pytest.mark.asyncio
async def test_save_longterm_memory_reversible(
    memory_manager: AgentMemoryManager,
) -> None:
    """Verify save_longterm_memory stores fact and marks status as saved."""
    saved = await save_longterm_memory(
        "U-100", "Customer speaks German", category="preference", memory_manager=memory_manager
    )
    assert saved["status"] == "saved"
    assert saved["user_id"] == "U-100"
    assert "record_id" in saved


@pytest.mark.asyncio
async def test_update_ticket_status_with_audit_trail(
    sample_tickets: dict[str, SupportTicket],
    audit_log: list[ReversalAuditRecord],
) -> None:
    """Verify update_ticket_status updates state and appends to reversal audit log."""
    upd = await update_ticket_status(
        "T-101",
        "in_progress",
        comment="Assigned to support engineer",
        ticket_store=sample_tickets,
        audit_log=audit_log,
    )
    assert upd["previous_status"] == "open"
    assert upd["current_status"] == "in_progress"
    assert upd["reversible"] is True
    assert len(audit_log) == 1
    assert audit_log[0].target_id == "T-101"
    assert audit_log[0].previous_state["status"] == "open"


@pytest.mark.asyncio
async def test_escalate_to_tier2(
    sample_tickets: dict[str, SupportTicket],
    audit_log: list[ReversalAuditRecord],
) -> None:
    """Verify escalate_to_tier2 updates assigned team and records reversal audit."""
    esc = await escalate_to_tier2(
        "T-101",
        "Kernel panic on boot",
        urgency="urgent",
        ticket_store=sample_tickets,
        audit_log=audit_log,
    )
    assert esc["status"] == "escalated_tier2"
    assert esc["assigned_team"] == "tier2_engineering"
    assert esc["urgency"] == "urgent"
    assert sample_tickets["T-101"].priority == "urgent"
    assert esc["reversible"] is True
    assert len(audit_log) == 1
