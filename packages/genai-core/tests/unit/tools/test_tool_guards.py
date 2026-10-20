from __future__ import annotations

import pytest

from genai_core.models import ToolCall
from genai_core.tools import (
    ApprovalDecision,
    ApprovalRequest,
    InMemoryApprovalStore,
    LoopDetector,
    RiskLevel,
    ToolExecutionStatus,
    ToolExecutor,
    ToolRegistry,
    tool,
)


def test_loop_detector_consecutive_identical_calls():
    detector = LoopDetector(max_repeated_calls=3)

    # 1st call
    assert not detector.record_and_check("search", {"q": "cats"})
    # 2nd call
    assert not detector.record_and_check("search", {"q": "cats"})
    # 3rd identical call triggers loop!
    assert detector.record_and_check("search", {"q": "cats"})


def test_loop_detector_different_arguments_do_not_trigger():
    detector = LoopDetector(max_repeated_calls=3)

    assert not detector.record_and_check("search", {"q": "cats"})
    assert not detector.record_and_check("search", {"q": "dogs"})
    assert not detector.record_and_check("search", {"q": "birds"})
    assert not detector.record_and_check("search", {"q": "cats"})


def test_loop_detector_key_ordering_invariance():
    detector = LoopDetector(max_repeated_calls=2)

    # Dict keys in different orders decode to same canonical hash
    assert not detector.record_and_check("calc", {"a": 1, "b": 2})
    assert detector.record_and_check("calc", {"b": 2, "a": 1})


def test_loop_detector_reset():
    detector = LoopDetector(max_repeated_calls=2)
    assert not detector.record_and_check("search", {"q": "test"})
    detector.reset()
    assert not detector.record_and_check("search", {"q": "test"})


@pytest.mark.asyncio
async def test_in_memory_approval_store():
    store = InMemoryApprovalStore()
    req = ApprovalRequest(
        request_id="req-1",
        call_id="call-1",
        tool_name="delete_account",
        arguments={"user_id": "u-42"},
        risk_level=RiskLevel.IRREVERSIBLE,
        reason="Requires admin approval",
    )

    await store.save_request(req)
    fetched = await store.get_request("req-1")
    assert fetched is not None
    assert fetched.tool_name == "delete_account"

    pending = await store.list_pending_requests()
    assert len(pending) == 1
    assert pending[0].request_id == "req-1"

    decision = ApprovalDecision(approved=True, reason="Admin verified identity")
    await store.record_decision("req-1", decision)

    stored_dec = await store.get_decision("req-1")
    assert stored_dec is not None
    assert stored_dec.approved is True

    pending_after = await store.list_pending_requests()
    assert len(pending_after) == 0


@pytest.mark.asyncio
async def test_executor_approval_gating():
    registry = ToolRegistry()

    @tool(name="drop_table", risk_level=RiskLevel.IRREVERSIBLE)
    def drop_table(table_name: str) -> str:
        return f"Dropped {table_name}"

    registry.register(drop_table)
    store = InMemoryApprovalStore()
    executor = ToolExecutor(
        registry=registry,
        require_approval_for={RiskLevel.IRREVERSIBLE},
        approval_store=store,
    )

    call = ToolCall(id="c_danger", name="drop_table", arguments={"table_name": "orders"})
    res = await executor.execute(call)

    assert not res.is_success()
    assert res.status == ToolExecutionStatus.APPROVAL_REQUIRED
    assert "paused: requires approval" in (res.error_message or "")

    pending = await store.list_pending_requests()
    assert len(pending) == 1
    assert pending[0].tool_name == "drop_table"
    assert pending[0].arguments == {"table_name": "orders"}

    # Execute with bypass_approval=True
    res_approved = await executor.execute(call, bypass_approval=True)
    assert res_approved.is_success()
    assert res_approved.content == "Dropped orders"
