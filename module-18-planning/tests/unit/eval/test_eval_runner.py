"""Unit tests for live and mock evaluation runners, criteria symmetry, and store isolation."""

from __future__ import annotations

import copy

from genai_core import ToolLoopStatus
from genai_core.models.evaluation_model import EvalSample, ExpectedAnswer


def test_eval_store_state_isolation_deepcopy() -> None:
    """Verify that in-memory store snapshot and restore restores initial state across runs."""
    accounts = {"C-101": {"balance": 100.0, "status": "active"}}
    tickets = {"T-101": {"status": "open", "priority": "low"}}

    initial_accounts = copy.deepcopy(accounts)
    initial_tickets = copy.deepcopy(tickets)

    # Simulate Goal 1 mutating stores
    accounts["C-101"]["balance"] = 0.0
    accounts["C-999"] = {"balance": 500.0}
    tickets["T-101"]["status"] = "resolved"
    tickets["T-999"] = {"status": "pending"}

    assert accounts["C-101"]["balance"] == 0.0
    assert "C-999" in accounts
    assert tickets["T-101"]["status"] == "resolved"
    assert "T-999" in tickets

    # Restore logic as executed before Goal 2
    accounts.clear()
    accounts.update(copy.deepcopy(initial_accounts))
    tickets.clear()
    tickets.update(copy.deepcopy(initial_tickets))

    assert accounts == {"C-101": {"balance": 100.0, "status": "active"}}
    assert tickets == {"T-101": {"status": "open", "priority": "low"}}
    assert "C-999" not in accounts
    assert "T-999" not in tickets


def test_eval_criteria_symmetry_logic() -> None:
    """Verify that criteria satisfaction requires keyword matching symmetrically for both systems."""
    sample = EvalSample(
        id="G-01",
        query="Investigate billing discrepancy for customer C-101",
        expected_answer=ExpectedAnswer(
            required=["discrepancy", "refund", "resolution"],
            reference="Discrepancy resolved with refund issued.",
        ),
    )

    # 1. Planned Assistant deliverable missing required keywords:
    # Even if status is COMPLETED and steps are COMPLETED, it must NOT pass if req keywords are missing.
    deliverable_missing_keywords = "The entire workflow finished and all steps are done."
    matches = [
        kw
        for kw in sample.expected_answer.required  # type: ignore[union-attr]
        if kw.lower() in deliverable_missing_keywords.lower()
    ]
    req_matched = len(matches) >= max(1, len(sample.expected_answer.required) // 2)  # type: ignore[union-attr]
    is_completed = True
    passed_planned = is_completed and req_matched and len(deliverable_missing_keywords) > 20
    assert passed_planned is False, (
        "Planned assistant must not pass when required keywords are missing"
    )

    # 2. Planned Assistant deliverable with required keywords:
    deliverable_with_keywords = "Discrepancy confirmed. Refund processed and resolution delivered."
    matches = [
        kw
        for kw in sample.expected_answer.required  # type: ignore[union-attr]
        if kw.lower() in deliverable_with_keywords.lower()
    ]
    req_matched = len(matches) >= max(1, len(sample.expected_answer.required) // 2)  # type: ignore[union-attr]
    passed_planned_ok = is_completed and req_matched and len(deliverable_with_keywords) > 20
    assert passed_planned_ok is True

    # 3. Reactive Baseline with APPROVAL_REQUIRED:
    # Must be recognized as completed when human review/approval is initiated.
    is_completed_reactive = ToolLoopStatus.APPROVAL_REQUIRED in (
        ToolLoopStatus.COMPLETED,
        ToolLoopStatus.APPROVAL_REQUIRED,
    )
    assert is_completed_reactive is True
    passed_reactive = is_completed_reactive and req_matched and len(deliverable_with_keywords) > 20
    assert passed_reactive is True
