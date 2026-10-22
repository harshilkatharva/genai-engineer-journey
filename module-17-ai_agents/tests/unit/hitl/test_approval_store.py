"""Unit tests for PostgresApprovalStore in-memory and state management operations."""

from __future__ import annotations

import pytest
from genai_core.tools.models import ApprovalDecision, ApprovalRequest, RiskLevel

from research_and_support_agent.hitl.postgres_store import PostgresApprovalStore


@pytest.mark.asyncio
async def test_approval_store_save_and_retrieve() -> None:
    """Verify saving an ApprovalRequest and retrieving it by request_id."""
    store = PostgresApprovalStore()  # In-memory mode
    req = ApprovalRequest(
        request_id="req_test_01",
        call_id="call_test_01",
        tool_name="execute_refund",
        arguments={"customer_id": "C-101", "amount_usd": 75.0},
        risk_level=RiskLevel.IRREVERSIBLE,
    )

    await store.save_request(req)
    retrieved = await store.get_request("req_test_01")

    assert retrieved is not None
    assert retrieved.request_id == "req_test_01"
    assert retrieved.tool_name == "execute_refund"
    assert retrieved.arguments["amount_usd"] == 75.0


@pytest.mark.asyncio
async def test_approval_store_record_decision() -> None:
    """Verify recording a decision updates request status and stores the decision record."""
    store = PostgresApprovalStore()
    req = ApprovalRequest(
        request_id="req_test_02",
        call_id="call_test_02",
        tool_name="purge_expired",
        arguments={},
        risk_level=RiskLevel.IRREVERSIBLE,
    )
    await store.save_request(req)

    decision = ApprovalDecision(
        approved=True,
        reason="Approved by supervisor for quarterly maintenance",
    )
    await store.record_decision("req_test_02", decision)

    stored_decision = await store.get_decision("req_test_02")
    assert stored_decision is not None
    assert stored_decision.approved is True
    assert "quarterly maintenance" in stored_decision.reason


@pytest.mark.asyncio
async def test_approval_store_list_pending() -> None:
    """Verify listing pending requests returns only unapproved requests."""
    store = PostgresApprovalStore()
    req1 = ApprovalRequest(
        request_id="req_p1",
        call_id="c_1",
        tool_name="delete_longterm_memory_all",
        arguments={"user_id": "U-1"},
        risk_level=RiskLevel.IRREVERSIBLE,
    )
    req2 = ApprovalRequest(
        request_id="req_p2",
        call_id="c_2",
        tool_name="delete_episodic_memory_all",
        arguments={"user_id": "U-2"},
        risk_level=RiskLevel.IRREVERSIBLE,
    )
    await store.save_request(req1)
    await store.save_request(req2)

    # Approve req1
    await store.record_decision("req_p1", ApprovalDecision(approved=True))

    pending = await store.list_pending()
    assert len(pending) == 1
    assert pending[0].request_id == "req_p2"


@pytest.mark.asyncio
async def test_approval_store_non_existent_request() -> None:
    """Verify retrieving a non-existent request ID returns None."""
    store = PostgresApprovalStore()
    res = await store.get_request("non_existent_id")
    assert res is None
