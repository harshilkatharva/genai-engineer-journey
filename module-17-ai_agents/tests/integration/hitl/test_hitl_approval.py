"""Integration tests for Human-in-the-Loop pause, persistence, and resumption."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from genai_core import (
    ApprovalDecision,
    LLMManagerResponse,
    ToolCall,
    ToolLoopStatus,
)

from research_and_support_agent.agent import BoundedAgent


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hitl_pause_and_resume_with_modified_arguments(agent: BoundedAgent) -> None:
    """Verify execution pauses for irreversible tool, persists request, and resumes upon approval."""
    # Step 1: Model requests execute_refund for $50.00
    call = ToolCall(
        id="call_ref_1",
        name="execute_refund",
        arguments={"customer_id": "C-101", "amount_usd": 50.0, "reason": "damaged item"},
    )
    resp1 = LLMManagerResponse(
        text=None,
        model="gemini-1.5-flash",
        tool_calls=[call],
        usage={"input_tokens": 40, "output_tokens": 20},
    )
    # Step 2: After resume, model provides final confirmation
    resp2 = LLMManagerResponse(
        text="Refund has been successfully processed for $40.00.",
        model="gemini-1.5-flash",
        tool_calls=[],
        usage={"input_tokens": 50, "output_tokens": 15},
    )
    agent.llm_service.complete_tool = AsyncMock(side_effect=[resp1, resp2])

    # Initial Run -> Expect APPROVAL_REQUIRED
    initial_res = await agent.run(query="Issue $50 refund to C-101", user_id="U-1")
    assert initial_res.status == ToolLoopStatus.APPROVAL_REQUIRED
    assert initial_res.pending_approval is not None

    req = initial_res.pending_approval
    assert req.tool_name == "execute_refund"
    assert req.arguments["amount_usd"] == 50.0

    # Verify request was saved in approval store
    saved_req = await agent.approval_store.get_request(req.request_id)
    assert saved_req is not None
    assert saved_req.request_id == req.request_id

    # Step 3: Human operator reviews and approves with modified amount ($40.00)
    decision = ApprovalDecision(
        approved=True,
        reason="Partial refund authorized for minor damage",
        modified_arguments={"customer_id": "C-101", "amount_usd": 40.0, "reason": "damaged item"},
    )

    resumed_res = await agent.resume(
        request_id=req.request_id,
        decision=decision,
        actor_id="supervisor_alice",
        actor_role="support_team",
    )

    # Assert execution completed cleanly
    assert resumed_res.status == ToolLoopStatus.COMPLETED
    assert "Refund has been successfully processed" in (resumed_res.final_text or "")
    # Balance should be updated by $40.00 (from 100.0 to 140.0)
    assert agent.account_store["C-101"].balance_usd == 140.0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hitl_rejection_flow(agent: BoundedAgent) -> None:
    """Verify execution pauses and handles operator rejection gracefully."""
    call = ToolCall(
        id="call_ref_2",
        name="execute_refund",
        arguments={"customer_id": "C-101", "amount_usd": 100.0, "reason": "customer complaint"},
    )
    resp1 = LLMManagerResponse(
        text=None,
        model="gemini-1.5-flash",
        tool_calls=[call],
        usage={"input_tokens": 30, "output_tokens": 15},
    )
    resp2 = LLMManagerResponse(
        text="I understand the refund was declined by management. I have notified the customer.",
        model="gemini-1.5-flash",
        tool_calls=[],
        usage={"input_tokens": 40, "output_tokens": 20},
    )
    agent.llm_service.complete_tool = AsyncMock(side_effect=[resp1, resp2])

    # Pause
    initial_res = await agent.run(query="Issue $100 refund to C-101", user_id="U-1")
    assert initial_res.status == ToolLoopStatus.APPROVAL_REQUIRED
    req = initial_res.pending_approval
    assert req is not None

    # Operator rejects
    decision = ApprovalDecision(approved=False, reason="Outside return policy window")
    resumed_res = await agent.resume(
        request_id=req.request_id,
        decision=decision,
        actor_id="supervisor_alice",
        actor_role="support_team",
    )

    assert resumed_res.status == ToolLoopStatus.COMPLETED
    assert "declined" in (resumed_res.final_text or "").lower()
    # Balance unchanged
    assert agent.account_store["C-101"].balance_usd == 100.0
