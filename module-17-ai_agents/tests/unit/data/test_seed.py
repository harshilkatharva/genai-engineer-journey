"""Unit tests for sample data definitions and seeding structures."""

from __future__ import annotations

from research_and_support_agent.data.seed import (
    SAMPLE_CUSTOMERS,
    SAMPLE_KB_DOCUMENTS,
    SAMPLE_TICKETS,
)


def test_sample_customers_integrity() -> None:
    """Verify seeded customer accounts contain all required benchmark test accounts."""
    customer_ids = {c.customer_id for c in SAMPLE_CUSTOMERS}

    assert "C-101" in customer_ids  # Standard / Gold query target
    assert "C-102" in customer_ids  # Standard tier
    assert "C-103" in customer_ids  # VIP refund candidate
    assert "C-104" in customer_ids  # Flagged risk account

    # Check risk account specifics
    risk_acc = next(c for c in SAMPLE_CUSTOMERS if c.customer_id == "C-104")
    assert risk_acc.risk_flag is True


def test_sample_tickets_integrity() -> None:
    """Verify seeded support tickets have valid categories and statuses."""
    ticket_ids = {t.ticket_id for t in SAMPLE_TICKETS}

    assert "T-101" in ticket_ids  # Open ticket
    assert "T-102" in ticket_ids  # Hardware ticket
    assert "T-103" in ticket_ids  # Urgent engineering ticket

    t101 = next(t for t in SAMPLE_TICKETS if t.ticket_id == "T-101")
    assert t101.customer_id == "C-101"
    assert t101.status == "open"


def test_sample_kb_documents_integrity() -> None:
    """Verify sample knowledge base articles cover required benchmark policies."""
    assert len(SAMPLE_KB_DOCUMENTS) >= 3

    sources = {doc.metadata.get("source") for doc in SAMPLE_KB_DOCUMENTS if doc.metadata}
    assert "policy/warranty_policy.md" in sources
    assert "policy/refund_policy.md" in sources
    assert "policy/escalation_guide.md" in sources

    # Check warranty policy content
    warr_doc = next(
        doc
        for doc in SAMPLE_KB_DOCUMENTS
        if doc.metadata and doc.metadata.get("source") == "policy/warranty_policy.md"
    )
    assert "30-day warranty return window" in warr_doc.content
