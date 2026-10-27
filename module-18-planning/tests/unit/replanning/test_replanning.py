"""Unit tests for dynamic replanning engine and failure recovery."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from plan_and_execute_research_assistant.core.config import AssistantConfig
from plan_and_execute_research_assistant.models import (
    Plan,
    PlanStep,
    RunBudget,
    StepStatus,
)
from plan_and_execute_research_assistant.replanning.engine import (
    ReplanningEngine,
)
from plan_and_execute_research_assistant.replanning.exceptions import (
    ReplanningError,
    ReplanningLimitExceededError,
)
from plan_and_execute_research_assistant.replanning.prompts import (
    build_replanning_prompt,
)


@pytest.fixture
def sample_failed_plan() -> Plan:
    """Fixture providing a plan where a step has failed."""
    return Plan(
        goal="Investigate customer anomaly and process refund",
        version=1,
        steps=[
            PlanStep(
                id="step_1",
                description="Fetch customer profile",
                success_criteria="Profile extracted",
                status=StepStatus.COMPLETED,
                result_summary="Customer ID C-101 verified.",
            ),
            PlanStep(
                id="step_2",
                description="Query legacy transactions API",
                depends_on=["step_1"],
                success_criteria="Transaction records extracted",
                status=StepStatus.FAILED,
                error_message="HTTP 503: Legacy API endpoint deprecated",
            ),
            PlanStep(
                id="step_3",
                description="Calculate refund amount and apply credit",
                depends_on=["step_2"],
                success_criteria="Refund credited",
                status=StepStatus.PENDING,
            ),
        ],
    )


def test_build_replanning_prompt(sample_failed_plan: Plan) -> None:
    """Verify build_replanning_prompt correctly formats context, critique, and instructions."""
    completed_steps = [sample_failed_plan.steps[0]]
    failed_step = sample_failed_plan.steps[1]
    remaining_steps = [sample_failed_plan.steps[2]]

    prompt = build_replanning_prompt(
        goal=sample_failed_plan.goal,
        completed_steps=completed_steps,
        failed_step=failed_step,
        failure_reason="Legacy API unreachable; use database search instead.",
        remaining_steps=remaining_steps,
        capabilities={"db_search": "Direct database query"},
        user_preferences=["Prefer SQL database queries over legacy REST"],
    )

    assert sample_failed_plan.goal in prompt
    assert "step_1" in prompt
    assert "Customer ID C-101 verified." in prompt
    assert "step_2" in prompt
    assert "Legacy API unreachable" in prompt
    assert "step_3" in prompt
    assert "db_search" in prompt
    assert "Prefer SQL database queries" in prompt
    assert "CRITICAL WORK PRESERVATION INVARIANT" in prompt


@pytest.mark.asyncio
async def test_replan_limit_exceeded_raises(sample_failed_plan: Plan) -> None:
    """Verify ReplanningLimitExceededError is raised when replan ceiling is reached."""
    mock_llm = MagicMock()
    config = AssistantConfig(max_replans=3)
    engine = ReplanningEngine(llm_service=mock_llm, config=config)

    with pytest.raises(ReplanningLimitExceededError) as exc_info:
        await engine.replan(
            current_plan=sample_failed_plan,
            failed_step_id="step_2",
            failure_reason="Failed again",
            step_results={"step_1": "Customer C-101 profile verified."},
            replan_count=3,  # Already reached limit
        )

    assert "Replanning ceiling exceeded (3/3)" in str(exc_info.value)
    mock_llm.complete_structure.assert_not_called()


@pytest.mark.asyncio
async def test_replan_preserves_completed_work(sample_failed_plan: Plan) -> None:
    """Verify completed steps remain immutable with original IDs and cached results."""
    mock_llm = MagicMock()

    # The LLM generates a revised plan with a new alternative step replacing step_2
    revised_candidate = Plan(
        goal=sample_failed_plan.goal,
        steps=[
            PlanStep(
                id="step_1",
                description="Fetch customer profile",
                success_criteria="Profile extracted",
                status=StepStatus.PENDING,  # LLM mistakenly set to PENDING
            ),
            PlanStep(
                id="step_2_alt",
                description="Query Postgres database directly for transactions",
                depends_on=["step_1"],
                success_criteria="Transactions retrieved from database",
            ),
            PlanStep(
                id="step_3",
                description="Calculate refund and credit customer",
                depends_on=["step_2_alt"],
                success_criteria="Refund credited",
            ),
        ],
    )
    mock_llm.complete_structure = AsyncMock(return_value=revised_candidate)

    engine = ReplanningEngine(llm_service=mock_llm)
    result = await engine.replan(
        current_plan=sample_failed_plan,
        failed_step_id="step_2",
        failure_reason="Legacy API unavailable; switch to direct DB query",
        step_results={"step_1": "Customer ID C-101 verified."},
        replan_count=0,
    )

    new_plan = result.new_plan
    assert new_plan.version == 2
    assert new_plan.parent_version == 1
    assert new_plan.created_by == "replanner"
    assert "step_2" in str(new_plan.change_summary)

    # Work Preservation Invariant: step_1 must be strictly COMPLETED with cached output
    step_1 = next(s for s in new_plan.steps if s.id == "step_1")
    assert step_1.status == StepStatus.COMPLETED
    assert step_1.result_summary == "Customer ID C-101 verified."

    assert "step_1" in result.preserved_step_ids
    assert "step_2_alt" in result.new_step_ids
    assert "step_3" in result.new_step_ids
    assert result.replan_count == 1


@pytest.mark.asyncio
async def test_replan_preserves_settled_sibling_work() -> None:
    """Verify settled parallel siblings that completed during failure are preserved."""
    initial_plan = Plan(
        goal="Parallel investigation and action",
        version=1,
        steps=[
            PlanStep(id="root", description="Fetch account details", success_criteria="Done"),
            PlanStep(
                id="branch_a",
                description="Run credit score check",
                depends_on=["root"],
                success_criteria="Done",
            ),
            PlanStep(
                id="branch_b",
                description="Query fraud history database",
                depends_on=["root"],
                success_criteria="Done",
            ),
            PlanStep(
                id="decision",
                description="Synthesize risk profile",
                depends_on=["branch_a", "branch_b"],
                success_criteria="Done",
            ),
        ],
    )

    # Both root and branch_b completed successfully; branch_a failed
    step_results = {
        "root": "Account details: verified premium tier",
        "branch_b": "Fraud history: clean, 0 suspicious flags",
    }

    mock_llm = MagicMock()
    # LLM replaces branch_a with branch_a_remedial and updates decision step
    revised_candidate = Plan(
        goal=initial_plan.goal,
        steps=[
            PlanStep(id="root", description="Fetch account details", success_criteria="Done"),
            PlanStep(
                id="branch_b",
                description="Query fraud history database",
                depends_on=["root"],
                success_criteria="Done",
            ),
            PlanStep(
                id="branch_a_remedial",
                description="Manual bureau fallback for credit check",
                depends_on=["root"],
                success_criteria="Credit score obtained via fallback bureau",
            ),
            PlanStep(
                id="decision",
                description="Synthesize risk profile",
                depends_on=["branch_a_remedial", "branch_b"],
                success_criteria="Done",
            ),
        ],
    )
    mock_llm.complete_structure = AsyncMock(return_value=revised_candidate)

    engine = ReplanningEngine(llm_service=mock_llm)
    result = await engine.replan(
        current_plan=initial_plan,
        failed_step_id="branch_a",
        failure_reason="Credit bureau API rate limited",
        step_results=step_results,
        replan_count=0,
    )

    # Verify BOTH root and settled sibling branch_b are preserved
    preserved_ids = set(result.preserved_step_ids)
    assert "root" in preserved_ids
    assert "branch_b" in preserved_ids

    root_step = next(s for s in result.new_plan.steps if s.id == "root")
    branch_b_step = next(s for s in result.new_plan.steps if s.id == "branch_b")
    assert root_step.status == StepStatus.COMPLETED
    assert root_step.result_summary == "Account details: verified premium tier"
    assert branch_b_step.status == StepStatus.COMPLETED
    assert branch_b_step.result_summary == "Fraud history: clean, 0 suspicious flags"

    assert "branch_a_remedial" in result.new_step_ids


@pytest.mark.asyncio
async def test_replan_triggers_repairer_on_invalid_dag(sample_failed_plan: Plan) -> None:
    """Verify PlanRepairer is invoked if the LLM produces an invalid revised DAG."""
    mock_llm = MagicMock()
    mock_repairer = MagicMock()

    # Candidate plan with a cycle between step_2_alt and step_3
    invalid_candidate = Plan(
        goal=sample_failed_plan.goal,
        steps=[
            PlanStep(id="step_1", description="Fetch customer", success_criteria="Done"),
            PlanStep(
                id="step_2_alt",
                description="Process alt",
                depends_on=["step_3"],
                success_criteria="Done",
            ),
            PlanStep(
                id="step_3",
                description="Apply refund",
                depends_on=["step_2_alt"],
                success_criteria="Done",
            ),
        ],
    )

    repaired_valid_plan = Plan(
        goal=sample_failed_plan.goal,
        steps=[
            PlanStep(id="step_1", description="Fetch customer", success_criteria="Done"),
            PlanStep(
                id="step_2_alt",
                description="Process alt",
                depends_on=["step_1"],
                success_criteria="Done",
            ),
            PlanStep(
                id="step_3",
                description="Apply refund",
                depends_on=["step_2_alt"],
                success_criteria="Done",
            ),
        ],
    )

    mock_llm.complete_structure = AsyncMock(return_value=invalid_candidate)
    mock_repairer.repair_plan = AsyncMock(return_value=repaired_valid_plan)

    engine = ReplanningEngine(
        llm_service=mock_llm,
        repairer=mock_repairer,
    )

    result = await engine.replan(
        current_plan=sample_failed_plan,
        failed_step_id="step_2",
        failure_reason="API timeout",
        step_results={"step_1": "Verified C-101"},
        replan_count=1,
    )

    mock_repairer.repair_plan.assert_awaited_once()
    assert result.new_plan.version == 2
    assert result.replan_count == 2


@pytest.mark.asyncio
async def test_replan_enforces_budget_tracking(sample_failed_plan: Plan) -> None:
    """Verify RunBudget tracks token consumption and cost during replanning."""
    mock_llm = MagicMock()
    revised_candidate = Plan(
        goal=sample_failed_plan.goal,
        steps=[
            PlanStep(id="step_1", description="Fetch customer", success_criteria="Done"),
            PlanStep(
                id="step_2_alt",
                description="DB query",
                depends_on=["step_1"],
                success_criteria="Done",
            ),
        ],
    )

    mock_resp = MagicMock()
    mock_resp.structured_response = revised_candidate
    mock_resp.usage.total_tokens = 450
    mock_resp.cost_usd = 0.005
    mock_llm.complete_structure = AsyncMock(return_value=mock_resp)

    budget = RunBudget(max_total_tokens=10000, max_total_cost_usd=1.0)
    budget.start_timer()

    engine = ReplanningEngine(llm_service=mock_llm)
    await engine.replan(
        current_plan=sample_failed_plan,
        failed_step_id="step_2",
        failure_reason="API crashed",
        step_results={"step_1": "Verified"},
        replan_count=0,
        budget=budget,
    )

    assert budget.current_tokens == 450
    assert budget.current_cost_usd == pytest.approx(0.005)


@pytest.mark.asyncio
async def test_replan_llm_failure_raises_replanning_error(sample_failed_plan: Plan) -> None:
    """Verify ReplanningError is raised when LLM invocation throws an unhandled error."""
    mock_llm = MagicMock()
    mock_llm.complete_structure = AsyncMock(side_effect=RuntimeError("Provider network down"))

    engine = ReplanningEngine(llm_service=mock_llm)

    with pytest.raises(ReplanningError) as exc_info:
        await engine.replan(
            current_plan=sample_failed_plan,
            failed_step_id="step_2",
            failure_reason="Network error",
            step_results={"step_1": "Verified"},
            replan_count=0,
        )

    assert "Replanning LLM invocation failed" in str(exc_info.value)
