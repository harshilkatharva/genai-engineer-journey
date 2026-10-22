"""Unit tests for PostgreSQL storage helpers."""

from __future__ import annotations

import json

from research_and_support_agent.models import CustomerAccount, SupportTicket, TicketStatus


def test_customer_json_serialization() -> None:
    """Verify customer active tickets and metadata serialize and deserialize accurately."""
    acc = CustomerAccount(
        customer_id="C-101",
        name="Alice Smith",
        email="alice@example.com",
        tier="Gold",
        balance_usd=1250.0,
        active_tickets=["T-1001", "T-1002"],
        risk_flag=False,
        metadata={"preferred_contact": "email", "assigned_rep": "REP-42"},
    )

    # Simulate DB serialization
    serialized_tickets = json.dumps(acc.active_tickets)
    serialized_meta = json.dumps(acc.metadata)

    # Simulate DB read
    recovered_tickets = json.loads(serialized_tickets)
    recovered_meta = json.loads(serialized_meta)

    recovered = CustomerAccount(
        customer_id=acc.customer_id,
        name=acc.name,
        email=acc.email,
        tier=acc.tier,
        balance_usd=acc.balance_usd,
        active_tickets=recovered_tickets,
        risk_flag=acc.risk_flag,
        metadata=recovered_meta,
    )

    assert recovered.customer_id == "C-101"
    assert recovered.active_tickets == ["T-1001", "T-1002"]
    assert recovered.metadata["assigned_rep"] == "REP-42"


def test_ticket_history_serialization() -> None:
    """Verify support ticket status and history list preservation."""
    ticket = SupportTicket(
        ticket_id="T-1001",
        customer_id="C-101",
        title="Payment inquiry",
        description="Invoice mismatch on order #42",
        status=TicketStatus.OPEN,
        history=[
            {"action": "created", "timestamp": 1700000000.0, "note": "Portal submission"},
            {"action": "assigned", "timestamp": 1700000100.0, "note": "Assigned to tier1"},
        ],
    )

    serialized_history = json.dumps(ticket.history)
    recovered_history = json.loads(serialized_history)

    recovered = SupportTicket(
        ticket_id=ticket.ticket_id,
        customer_id=ticket.customer_id,
        title=ticket.title,
        description=ticket.description,
        status=TicketStatus("open"),
        history=recovered_history,
        created_at=ticket.created_at,
        updated_at=ticket.updated_at,
    )

    assert recovered.ticket_id == "T-1001"
    assert recovered.status == TicketStatus.OPEN
    assert len(recovered.history) == 2
    assert recovered.history[0]["action"] == "created"
