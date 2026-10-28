"""Unit tests for HumanPlanReviewGateway and immutable plan versioning."""

from __future__ import annotations

import pytest

from plan_and_execute_research_assistant.core import AssistantConfig
from plan_and_execute_research_assistant.models import (
    Plan,
    PlanReviewDecision,
    PlanReviewRequest,
    PlanStep,
    ReviewAction,
)
from plan_and_execute_research_assistant.review import (
    HumanPlanReviewGateway,
    PlanRejectedError,
    ReviewValidationError,
)


@pytest.fixture
def sample_plan() -> Plan:
    """Fixture providing a standard valid 2-step plan."""
    return Plan(
        goal="Audit user permissions",
        steps=[
            PlanStep(
                id="step_1",
                description="Fetch user roles from directory",
                tool_name_hint="directory_lookup",
                success_criteria="User roles retrieved",
            ),
            PlanStep(
                id="step_2",
                description="Compare roles against least privilege policy",
                depends_on=["step_1"],
                tool_name_hint="compliance_check",
                success_criteria="Discrepancies identified",
            ),
        ],
        version=1,
        created_by="planner_llm",
    )


def test_format_plan_for_review(sample_plan: Plan) -> None:
    """Verify format_plan_for_review generates human-readable representation."""
    gateway = HumanPlanReviewGateway()
    formatted = gateway.format_plan_for_review(sample_plan)

    assert "Audit user permissions" in formatted
    assert "[step_1]" in formatted
    assert "directory_lookup" in formatted
    assert "[step_2]" in formatted
    assert "Dependencies:    step_1" in formatted
    assert "User roles retrieved" in formatted


def test_create_review_request(sample_plan: Plan) -> None:
    """Verify create_review_request wraps plan and goal with valid UUID."""
    gateway = HumanPlanReviewGateway()
    req = gateway.create_review_request(
        plan=sample_plan,
        goal="Audit user permissions",
        user_id="user_admin",
    )

    assert isinstance(req, PlanReviewRequest)
    assert req.plan.goal == "Audit user permissions"
    assert req.user_id == "user_admin"
    assert len(req.request_id) > 10


def test_process_decision_approve(sample_plan: Plan) -> None:
    """Verify operator APPROVE decision returns the original plan unmodified."""
    gateway = HumanPlanReviewGateway()
    req = gateway.create_review_request(sample_plan, sample_plan.goal, "user_1")
    decision = PlanReviewDecision(
        action=ReviewAction.APPROVE,
        reviewer_id="lead_operator",
        reason="Looks solid",
    )

    approved = gateway.process_decision(req, decision)
    assert approved.version == 1
    assert approved.parent_version is None
    assert approved.created_by == "planner_llm"
    assert len(approved.steps) == 2


def test_process_decision_modify_success(sample_plan: Plan) -> None:
    """Verify operator MODIFY increments version and records parent and author."""
    gateway = HumanPlanReviewGateway()
    req = gateway.create_review_request(sample_plan, sample_plan.goal, "user_1")

    # Operator adds a 3rd step
    modified = sample_plan.model_copy(deep=True)
    modified.steps.append(
        PlanStep(
            id="step_3",
            description="Generate compliance audit PDF",
            depends_on=["step_2"],
            tool_name_hint="pdf_export",
            success_criteria="PDF generated",
        )
    )

    decision = PlanReviewDecision(
        action=ReviewAction.MODIFY,
        reviewer_id="lead_operator",
        reason="Added reporting step for leadership",
        modified_plan=modified,
    )

    result = gateway.process_decision(req, decision)
    assert result.version == 2
    assert result.parent_version == 1
    assert result.created_by == "human_reviewer"
    assert result.change_summary == "Added reporting step for leadership"
    assert len(result.steps) == 3


def test_process_decision_modify_validation_failure(sample_plan: Plan) -> None:
    """Verify operator MODIFY with an invalid DAG raises ReviewValidationError."""
    gateway = HumanPlanReviewGateway()
    req = gateway.create_review_request(sample_plan, sample_plan.goal, "user_1")

    # Operator creates a cycle: step_1 -> step_2 -> step_1
    modified = sample_plan.model_copy(deep=True)
    modified.steps[0].depends_on = ["step_2"]

    decision = PlanReviewDecision(
        action=ReviewAction.MODIFY,
        reviewer_id="operator_novice",
        reason="Accidental cycle",
        modified_plan=modified,
    )

    with pytest.raises(ReviewValidationError) as exc_info:
        gateway.process_decision(req, decision)

    assert "Operator-modified plan failed DAG validation" in str(exc_info.value)
    assert exc_info.value.validation_error is not None


def test_process_decision_modify_without_plan_raises(sample_plan: Plan) -> None:
    """Verify MODIFY action without modified_plan raises ValueError."""
    gateway = HumanPlanReviewGateway()
    req = gateway.create_review_request(sample_plan, sample_plan.goal, "user_1")
    decision = PlanReviewDecision(
        action=ReviewAction.MODIFY,
        reviewer_id="lead_operator",
        modified_plan=None,
    )

    with pytest.raises(ValueError, match="requires 'modified_plan'"):
        gateway.process_decision(req, decision)


def test_process_decision_reject(sample_plan: Plan) -> None:
    """Verify operator REJECT decision raises PlanRejectedError with details."""
    gateway = HumanPlanReviewGateway()
    req = gateway.create_review_request(sample_plan, sample_plan.goal, "user_1")
    decision = PlanReviewDecision(
        action=ReviewAction.REJECT,
        reviewer_id="lead_operator",
        reason="Strategy violates data privacy policy",
    )

    with pytest.raises(PlanRejectedError) as exc_info:
        gateway.process_decision(req, decision)

    assert exc_info.value.reviewer_id == "lead_operator"
    assert exc_info.value.reason == "Strategy violates data privacy policy"


@pytest.mark.asyncio
async def test_review_plan_auto_approve_bypass(sample_plan: Plan) -> None:
    """Verify auto_approve_review=True bypasses decision_provider."""
    config = AssistantConfig(auto_approve_review=True)
    gateway = HumanPlanReviewGateway(config=config)

    # decision_provider is None, should not raise
    approved = await gateway.review_plan(
        plan=sample_plan,
        goal="Audit user permissions",
        user_id="user_admin",
    )
    assert approved.version == 1
    assert approved.created_by == "planner_llm"


@pytest.mark.asyncio
async def test_review_plan_async_decision_provider(sample_plan: Plan) -> None:
    """Verify review_plan invokes async decision_provider when auto_approve is False."""
    config = AssistantConfig(auto_approve_review=False)
    gateway = HumanPlanReviewGateway(config=config)

    async def mock_async_provider(req: PlanReviewRequest) -> PlanReviewDecision:
        assert req.goal == "Audit user permissions"
        return PlanReviewDecision(
            action=ReviewAction.APPROVE,
            reviewer_id="async_operator",
            reason="Approved via async prompt",
        )

    approved = await gateway.review_plan(
        plan=sample_plan,
        goal="Audit user permissions",
        user_id="user_admin",
        decision_provider=mock_async_provider,
    )
    assert approved.version == 1


@pytest.mark.asyncio
async def test_review_plan_missing_decision_provider_raises(sample_plan: Plan) -> None:
    """Verify review_plan raises ValueError if provider is missing and auto_approve is False."""
    config = AssistantConfig(auto_approve_review=False)
    gateway = HumanPlanReviewGateway(config=config)

    with pytest.raises(ValueError, match="decision_provider callable is required"):
        await gateway.review_plan(
            plan=sample_plan,
            goal="Audit user permissions",
            user_id="user_admin",
            decision_provider=None,
        )


def test_process_decision_modify_enforces_deep_copy(sample_plan: Plan) -> None:
    """Verify that modifying stamped plan does not mutate the original decision modified_plan (PLN-007)."""
    gateway = HumanPlanReviewGateway()
    req = gateway.create_review_request(sample_plan, sample_plan.goal, "user_1")

    modified = sample_plan.model_copy(deep=True)
    decision = PlanReviewDecision(
        action=ReviewAction.MODIFY,
        reviewer_id="lead_operator",
        modified_plan=modified,
    )

    result = gateway.process_decision(req, decision)
    # Mutate the step in the returned stamped plan
    result.steps[0].description = "Mutated description after approval"

    # Verify that the input modified plan's step was NOT mutated
    assert modified.steps[0].description == "Fetch user roles from directory"


@pytest.mark.asyncio
async def test_review_plan_interactive_cli_approve(
    monkeypatch: pytest.MonkeyPatch, sample_plan: Plan
) -> None:
    """Verify interactive CLI review prompt auto-approves on 'a' input (PLN-010)."""
    import sys

    config = AssistantConfig(auto_approve_review=False)
    gateway = HumanPlanReviewGateway(config=config)

    # Simulate interactive TTY stdin
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "a")

    approved = await gateway.review_plan(
        plan=sample_plan,
        goal="Audit user permissions",
        user_id="user_admin",
        decision_provider=None,
    )
    assert approved.version == 1
    assert approved.created_by == "planner_llm"


@pytest.mark.asyncio
async def test_review_plan_interactive_cli_reject(
    monkeypatch: pytest.MonkeyPatch, sample_plan: Plan
) -> None:
    """Verify interactive CLI review prompt rejects on 'r' input (PLN-010)."""
    import sys

    config = AssistantConfig(auto_approve_review=False)
    gateway = HumanPlanReviewGateway(config=config)

    inputs = iter(["r", "Security vulnerability detected"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    with pytest.raises(PlanRejectedError, match="Security vulnerability detected"):
        await gateway.review_plan(
            plan=sample_plan,
            goal="Audit user permissions",
            user_id="user_admin",
            decision_provider=None,
        )
