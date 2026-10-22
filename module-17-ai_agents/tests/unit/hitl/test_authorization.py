"""Unit tests for human-in-the-loop authorization policy and argument revalidation."""

from __future__ import annotations

import pytest
from genai_core.exceptions import ToolValidationError
from genai_core.tools.models import ApprovalDecision, ApprovalRequest, RiskLevel

from research_and_support_agent.hitl.authorization import AuthorizationValidator


def test_memory_owner_authorization() -> None:
    """Verify memory deletion requires confirmation by the specific owner."""
    req = ApprovalRequest(
        call_id="c_1",
        tool_name="delete_longterm_memory_all",
        arguments={"user_id": "U-100"},
        risk_level=RiskLevel.IRREVERSIBLE,
    )
    decision = ApprovalDecision(approved=True)

    # Approved by owner -> PASS
    assert (
        AuthorizationValidator.validate_approval(
            request=req,
            decision=decision,
            actor_id="U-100",
            actor_role="user",
        )
        is True
    )

    # Approved by different user -> FAIL
    with pytest.raises(ToolValidationError, match="must be confirmed by the memory owner"):
        AuthorizationValidator.validate_approval(
            request=req,
            decision=decision,
            actor_id="U-IMPOSTER",
            actor_role="user",
        )


def test_support_team_authorization() -> None:
    """Verify refund and customer-data deletion requires support-team role."""
    req = ApprovalRequest(
        call_id="c_2",
        tool_name="execute_refund",
        arguments={"customer_id": "C-101", "amount_usd": 50.0},
        risk_level=RiskLevel.IRREVERSIBLE,
    )
    decision = ApprovalDecision(approved=True)

    # Approved by support_team -> PASS
    assert (
        AuthorizationValidator.validate_approval(
            request=req,
            decision=decision,
            actor_id="agent_99",
            actor_role="support_team",
        )
        is True
    )

    # Approved by regular customer -> FAIL
    with pytest.raises(ToolValidationError, match="requires authorization from 'support_team'"):
        AuthorizationValidator.validate_approval(
            request=req,
            decision=decision,
            actor_id="C-101",
            actor_role="customer",
        )


def test_llm_self_approval_blocked() -> None:
    """Verify that an automated agent or LLM cannot approve its own action."""
    req = ApprovalRequest(
        call_id="c_3",
        tool_name="execute_refund",
        arguments={"customer_id": "C-101", "amount_usd": 100.0},
        risk_level=RiskLevel.IRREVERSIBLE,
    )
    decision = ApprovalDecision(approved=True)

    with pytest.raises(ToolValidationError, match="prohibited from approving its own actions"):
        AuthorizationValidator.validate_approval(
            request=req,
            decision=decision,
            actor_id="agent_self",
            actor_role="llm",
        )


def test_argument_revalidation() -> None:
    """Verify that modified arguments are validated before execution."""
    # Valid refund arguments
    AuthorizationValidator.revalidate_arguments(
        "execute_refund", {"customer_id": "C-1", "amount_usd": 25.0}
    )

    # Negative refund amount -> FAIL
    with pytest.raises(ToolValidationError, match="positive number"):
        AuthorizationValidator.revalidate_arguments(
            "execute_refund", {"customer_id": "C-1", "amount_usd": -10.0}
        )

    # Missing customer_id -> FAIL
    with pytest.raises(ToolValidationError, match="requires a valid customer_id"):
        AuthorizationValidator.revalidate_arguments("execute_refund", {"amount_usd": 50.0})
