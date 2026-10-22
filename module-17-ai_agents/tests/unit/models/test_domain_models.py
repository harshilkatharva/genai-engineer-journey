"""Unit tests for domain models and state representations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_and_support_agent.models import (
    CustomerAccount,
    ReversalAuditRecord,
    SupportTicket,
    TicketStatus,
)


def test_customer_account_defaults() -> None:
    """Verify customer account defaults and field validation."""
    acc = CustomerAccount(
        customer_id="C-999",
        name="Charlie Test",
        email="charlie@example.com",
    )
    assert acc.customer_id == "C-999"
    assert acc.tier == "standard"
    assert acc.balance_usd == 0.0
    assert acc.active_tickets == []
    assert not acc.risk_flag
    assert acc.metadata == {}


def test_customer_account_custom_values() -> None:
    """Verify custom values and balance modifications."""
    acc = CustomerAccount(
        customer_id="C-101",
        name="Alice Smith",
        email="alice@example.com",
        tier="gold",
        balance_usd=150.0,
        active_tickets=["T-101", "T-102"],
        risk_flag=True,
        metadata={"priority_client": True},
    )
    assert acc.tier == "gold"
    assert acc.balance_usd == 150.0
    assert acc.risk_flag is True
    assert len(acc.active_tickets) == 2


def test_support_ticket_lifecycle() -> None:
    """Verify support ticket transitions and default properties."""
    t = SupportTicket(
        ticket_id="T-501",
        customer_id="C-101",
        title="Network lag",
        description="High ping observed on primary connection.",
    )
    assert t.status == TicketStatus.OPEN
    assert t.assigned_team == "tier1_support"
    assert t.priority == "medium"
    assert t.history == []

    # Update status to valid enum
    t.status = TicketStatus.IN_PROGRESS
    assert t.status == TicketStatus.IN_PROGRESS

    t.status = TicketStatus.ESCALATED_TIER2
    assert t.status == TicketStatus.ESCALATED_TIER2


def test_invalid_ticket_status() -> None:
    """Verify that assigning invalid status strings fails validation."""
    with pytest.raises(ValidationError):
        SupportTicket(
            ticket_id="T-999",
            customer_id="C-101",
            title="Broken item",
            description="Details",
            status="non_existent_status",  # type: ignore[arg-type]
        )


def test_reversal_audit_record() -> None:
    """Verify ReversalAuditRecord captures before and after states for rollback."""
    rec = ReversalAuditRecord(
        operation_id="op_123",
        tool_name="update_ticket_status",
        target_entity="support_ticket",
        target_id="T-101",
        previous_state={"status": "open"},
        new_state={"status": "in_progress"},
    )
    assert rec.operation_id == "op_123"
    assert rec.reverted is False
    assert rec.previous_state["status"] == "open"
    assert rec.new_state["status"] == "in_progress"
    assert rec.timestamp > 0.0
