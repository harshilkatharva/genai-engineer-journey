from __future__ import annotations

import time

import pytest

from plan_and_execute_research_assistant.models import (
    BudgetExceededError,
    Plan,
    PlanReviewDecision,
    PlanReviewRequest,
    PlanStep,
    PlanVersion,
    ReviewAction,
    RunBudget,
    RunRecord,
    RunStatus,
    StepExecutionRecord,
    StepJudgeVerdict,
    StepStatus,
    StepVerificationResult,
)


def test_step_status_values() -> None:
    """Verify all StepStatus enum values."""
    assert StepStatus.PENDING == "pending"
    assert StepStatus.READY == "ready"
    assert StepStatus.RUNNING == "running"
    assert StepStatus.COMPLETED == "completed"
    assert StepStatus.FAILED == "failed"
    assert StepStatus.BLOCKED == "blocked"
    assert StepStatus.SKIPPED == "skipped"


def test_run_status_values() -> None:
    """Verify all RunStatus enum values."""
    assert RunStatus.INITIALIZING == "initializing"
    assert RunStatus.PLANNING == "planning"
    assert RunStatus.REVIEW_PENDING == "review_pending"
    assert RunStatus.EXECUTING == "executing"
    assert RunStatus.REPLANNING == "replanning"
    assert RunStatus.COMPLETED == "completed"
    assert RunStatus.FAILED == "failed"
    assert RunStatus.CANCELLED == "cancelled"


def test_review_action_values() -> None:
    """Verify all ReviewAction enum values."""
    assert ReviewAction.APPROVE == "approve"
    assert ReviewAction.MODIFY == "modify"
    assert ReviewAction.REJECT == "reject"


def test_plan_step_defaults_and_validation() -> None:
    """Verify PlanStep initialization defaults and serialization."""
    step = PlanStep(
        id="step_1",
        description="Query PostgreSQL database for customer record",
        success_criteria="Customer record returned with tier and balance",
    )
    assert step.id == "step_1"
    assert step.description == "Query PostgreSQL database for customer record"
    assert step.tool_name_hint is None
    assert step.depends_on is None
    assert step.status == StepStatus.PENDING
    assert step.attempts == 0
    assert step.result_summary is None
    assert step.error_message is None

    # Test round-trip JSON serialization
    step_json = step.model_dump_json()
    reconstructed = PlanStep.model_validate_json(step_json)
    assert reconstructed.id == step.id
    assert reconstructed.status == StepStatus.PENDING


def test_plan_defaults_and_versioning() -> None:
    """Verify Plan model defaults, parent_version, and assumptions."""
    step_1 = PlanStep(id="s1", description="Step 1", success_criteria="Done")
    step_2 = PlanStep(id="s2", description="Step 2", depends_on=["s1"], success_criteria="Done")

    plan_v1 = Plan(
        goal="Perform automated customer refund",
        steps=[step_1, step_2],
        assumptions=["User is authenticated"],
    )
    assert plan_v1.version == 1
    assert plan_v1.parent_version is None
    assert plan_v1.created_by == "planner_llm"
    assert len(plan_v1.steps) == 2
    assert plan_v1.assumptions == ["User is authenticated"]

    # Version 2 derived from Version 1
    plan_v2 = Plan(
        goal=plan_v1.goal,
        steps=plan_v1.steps,
        version=2,
        parent_version=1,
        created_by="human_reviewer",
        change_summary="Modified step 2 criteria",
    )
    assert plan_v2.version == 2
    assert plan_v2.parent_version == 1
    assert plan_v2.created_by == "human_reviewer"
    assert plan_v2.change_summary == "Modified step 2 criteria"


def test_plan_version_snapshot() -> None:
    """Verify PlanVersion snapshot model."""
    plan = Plan(
        goal="Test Goal",
        steps=[PlanStep(id="s1", description="Step 1", success_criteria="Done")],
    )
    pv = PlanVersion(
        version=1,
        plan=plan,
        created_by="planner_llm",
    )
    assert pv.version == 1
    assert pv.plan.goal == "Test Goal"
    assert pv.created_by == "planner_llm"


def test_review_models() -> None:
    """Verify PlanReviewRequest and PlanReviewDecision serialization."""
    plan = Plan(
        goal="Test Review",
        steps=[PlanStep(id="s1", description="Step 1", success_criteria="Done")],
    )
    req = PlanReviewRequest(
        plan=plan,
        goal=plan.goal,
        user_id="user_test_1",
    )
    assert len(req.request_id) > 0
    assert req.user_id == "user_test_1"

    decision_approve = PlanReviewDecision(
        action=ReviewAction.APPROVE,
        reviewer_id="operator_1",
    )
    assert decision_approve.action == ReviewAction.APPROVE
    assert decision_approve.modified_plan is None

    decision_modify = PlanReviewDecision(
        action=ReviewAction.MODIFY,
        reviewer_id="operator_1",
        reason="Added validation step",
        modified_plan=plan,
    )
    assert decision_modify.action == ReviewAction.MODIFY
    assert decision_modify.modified_plan is not None


def test_verification_models() -> None:
    """Verify StepJudgeVerdict and StepVerificationResult."""
    verdict = StepJudgeVerdict(
        is_satisfied=True,
        score=0.95,
        reasoning="Output explicitly addresses all criteria",
        missing_criteria=[],
    )
    assert verdict.is_satisfied is True
    assert verdict.score == 0.95

    res = StepVerificationResult(
        step_id="step_1",
        passed=True,
        programmatic_passed=True,
        model_passed=True,
        score=0.95,
        reasoning="All checks green",
    )
    assert res.passed is True
    assert res.step_id == "step_1"
    assert res.missing_criteria == []


def test_run_budget_lifecycle_and_checks() -> None:
    """Verify RunBudget timers, limits, and BudgetExceededError."""
    budget = RunBudget(
        max_total_tokens=1000,
        max_total_cost_usd=0.05,
        total_timeout_seconds=2.0,
    )
    assert budget.is_exhausted() is False
    assert budget.start_monotonic is None

    # Start timer
    budget.start_timer()
    assert budget.start_monotonic is not None
    assert budget.deadline_monotonic is not None
    assert budget.remaining_seconds() > 0.0

    # Record token usage under ceiling
    budget.record_usage(tokens=500, cost_usd=0.02)
    assert budget.current_tokens == 500
    assert budget.current_cost_usd == 0.02
    assert budget.is_token_exhausted() is False
    assert budget.is_cost_exhausted() is False
    budget.check_budget()  # Should not raise

    # Token breach
    budget.record_usage(tokens=600, cost_usd=0.01)
    assert budget.current_tokens == 1100
    assert budget.is_token_exhausted() is True
    assert budget.is_exhausted() is True
    with pytest.raises(BudgetExceededError, match="Token budget exhausted"):
        budget.check_budget()

    # Cost breach
    budget_cost = RunBudget(max_total_cost_usd=0.01)
    budget_cost.record_usage(tokens=10, cost_usd=0.02)
    assert budget_cost.is_cost_exhausted() is True
    with pytest.raises(BudgetExceededError, match="Cost budget exhausted"):
        budget_cost.check_budget()

    # Timeout breach
    budget_timeout = RunBudget(total_timeout_seconds=0.01)
    budget_timeout.start_timer()
    time.sleep(0.02)
    assert budget_timeout.is_timeout_exhausted() is True
    assert budget_timeout.remaining_seconds() == 0.0
    with pytest.raises(BudgetExceededError, match="Run deadline exceeded"):
        budget_timeout.check_budget()


def test_run_record_and_step_execution_record() -> None:
    """Verify StepExecutionRecord and RunRecord models."""
    step_rec = StepExecutionRecord(
        step_id="step_1",
        plan_version=1,
        status=StepStatus.COMPLETED,
        attempts=1,
        depends_on=[],
        tokens_used=120,
        cost_usd=0.0005,
        duration_ms=450.0,
        result_summary="Retrieved account info",
    )
    assert step_rec.step_id == "step_1"
    assert step_rec.status == StepStatus.COMPLETED

    run_rec = RunRecord(
        goal="Audit support tickets",
        user_id="user_42",
        status=RunStatus.EXECUTING,
        step_executions={"step_1": [step_rec]},
        total_tokens=120,
        total_cost_usd=0.0005,
    )
    assert run_rec.user_id == "user_42"
    assert len(run_rec.run_id) > 0
    assert "step_1" in run_rec.step_executions

    # Round trip JSON dump
    data = run_rec.model_dump_json()
    deserialized = RunRecord.model_validate_json(data)
    assert deserialized.goal == "Audit support tickets"
    assert deserialized.step_executions["step_1"][0].result_summary == "Retrieved account info"
