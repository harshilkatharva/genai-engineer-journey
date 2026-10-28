"""Unit tests for PlanAndExecuteAssistant orchestrator facade."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from plan_and_execute_research_assistant.core.config import AssistantConfig
from plan_and_execute_research_assistant.models import (
    Plan,
    PlanReviewDecision,
    PlanReviewRequest,
    PlanStep,
    ReviewAction,
    RunBudget,
    RunStatus,
    StepExecutionRecord,
    StepStatus,
    StepVerificationResult,
)
from plan_and_execute_research_assistant.orchestration.assistant import (
    PlanAndExecuteAssistant,
)
from plan_and_execute_research_assistant.protocols.reflector import (
    ReflectionResult,
)
from plan_and_execute_research_assistant.replanning.engine import ReplanResult
from plan_and_execute_research_assistant.scheduling.scheduler import (
    SchedulerExecutionResult,
)


@pytest.fixture
def mock_llm_service() -> MagicMock:
    """Mock LLMService for structured and unstructured generation."""
    mock = MagicMock()
    mock.complete = AsyncMock(return_value="Synthesized comprehensive deliverable.")
    mock.complete_structure = AsyncMock()
    return mock


@pytest.fixture
def sample_valid_plan() -> Plan:
    """Fixture providing a standard valid 2-step plan."""
    return Plan(
        goal="Investigate customer anomaly and credit account",
        version=1,
        steps=[
            PlanStep(
                id="step_1",
                description="Fetch customer information",
                success_criteria="Customer info verified",
            ),
            PlanStep(
                id="step_2",
                description="Apply refund credit",
                depends_on=["step_1"],
                success_criteria="Credit applied",
            ),
        ],
    )


@pytest.mark.asyncio
async def test_assistant_happy_path_e2e(
    mock_llm_service: MagicMock,
    sample_valid_plan: Plan,
) -> None:
    """Verify complete 5-phase lifecycle under happy-path execution."""
    mock_executor = MagicMock()
    mock_executor.get_available_capabilities = MagicMock(
        return_value={"tool_a": "Tool A description"}
    )

    mock_mem = MagicMock()
    mock_mem.get_user_preferences = AsyncMock(return_value=["Format as markdown"])
    mock_mem.record_plan_episode = AsyncMock()

    mock_store = MagicMock()
    mock_store.save_run = AsyncMock()
    mock_store._save_run_sync = MagicMock()

    mock_generator = MagicMock()
    mock_generator.generate_plan = AsyncMock(return_value=sample_valid_plan)

    step_rec = StepExecutionRecord(
        step_id="step_1",
        plan_version=1,
        status=StepStatus.COMPLETED,
        result_summary="Customer profile C-101 verified.",
        verification=StepVerificationResult(
            step_id="step_1",
            passed=True,
            programmatic_passed=True,
            model_passed=True,
            score=0.95,
            reasoning="Passed all checks",
        ),
    )

    mock_scheduler = MagicMock()
    mock_scheduler.execute_plan = AsyncMock(
        return_value=SchedulerExecutionResult(
            status=StepStatus.COMPLETED,
            step_results={
                "step_1": "Customer C-101 verified.",
                "step_2": "Refund $50 applied successfully.",
            },
            step_records={"step_1": [step_rec]},
        )
    )

    config = AssistantConfig(auto_approve_review=True)
    assistant = PlanAndExecuteAssistant(
        llm_service=mock_llm_service,
        executor=mock_executor,
        memory_manager=mock_mem,
        store=mock_store,
        plan_generator=mock_generator,
        scheduler=mock_scheduler,
        config=config,
    )

    result = await assistant.run(
        goal="Investigate customer anomaly and credit account",
        user_id="user_test_42",
    )

    assert result.status == RunStatus.COMPLETED
    assert result.final_deliverable == "Synthesized comprehensive deliverable."
    assert result.plan.version == 1
    assert result.total_tokens >= 0
    assert result.total_cost_usd >= 0.0

    mock_mem.record_plan_episode.assert_awaited_once()
    mock_store.save_run.assert_awaited()


@pytest.mark.asyncio
async def test_assistant_human_review_modify_flow(
    mock_llm_service: MagicMock,
    sample_valid_plan: Plan,
) -> None:
    """Verify human reviewer modification increments plan version and executes modified DAG."""
    mock_executor = MagicMock()
    mock_executor.get_available_capabilities = MagicMock(return_value={})

    mock_mem = MagicMock()
    mock_mem.get_user_preferences = AsyncMock(return_value=[])
    mock_mem.record_plan_episode = AsyncMock()

    mock_store = MagicMock()
    mock_store.save_run = AsyncMock()
    mock_store._save_run_sync = MagicMock()

    mock_generator = MagicMock()
    mock_generator.generate_plan = AsyncMock(return_value=sample_valid_plan)

    # Review decision provider modifying the plan
    modified_plan = sample_valid_plan.model_copy()
    modified_plan.steps.append(
        PlanStep(
            id="step_3_audit",
            description="Send audit receipt to customer",
            depends_on=["step_2"],
            success_criteria="Receipt emailed",
        )
    )

    async def mock_decision_provider(req: PlanReviewRequest) -> PlanReviewDecision:
        return PlanReviewDecision(
            action=ReviewAction.MODIFY,
            reviewer_id="human_reviewer_1",
            modified_plan=modified_plan,
            reason="Add audit receipt step",
        )

    mock_scheduler = MagicMock()
    mock_scheduler.execute_plan = AsyncMock(
        return_value=SchedulerExecutionResult(
            status=StepStatus.COMPLETED,
            step_results={"step_1": "Done", "step_2": "Done", "step_3_audit": "Emailed"},
        )
    )

    config = AssistantConfig(auto_approve_review=False)
    assistant = PlanAndExecuteAssistant(
        llm_service=mock_llm_service,
        executor=mock_executor,
        memory_manager=mock_mem,
        store=mock_store,
        plan_generator=mock_generator,
        scheduler=mock_scheduler,
        config=config,
    )

    result = await assistant.run(
        goal="Investigate and credit",
        user_id="user_admin",
        decision_provider=mock_decision_provider,
    )

    assert result.status == RunStatus.COMPLETED
    assert result.plan.version == 2
    assert result.plan.parent_version == 1
    assert result.plan.created_by == "human_reviewer"


@pytest.mark.asyncio
async def test_assistant_human_review_reject_flow(
    mock_llm_service: MagicMock,
    sample_valid_plan: Plan,
) -> None:
    """Verify human reviewer rejection transitions run to CANCELLED and halts."""
    mock_executor = MagicMock()
    mock_executor.get_available_capabilities = MagicMock(return_value={})

    mock_mem = MagicMock()
    mock_mem.get_user_preferences = AsyncMock(return_value=[])

    mock_store = MagicMock()
    mock_store.save_run = AsyncMock()
    mock_store._save_run_sync = MagicMock()

    mock_generator = MagicMock()
    mock_generator.generate_plan = AsyncMock(return_value=sample_valid_plan)

    async def mock_decision_provider(req: PlanReviewRequest) -> PlanReviewDecision:
        return PlanReviewDecision(
            action=ReviewAction.REJECT,
            reviewer_id="human_reviewer_1",
            reason="Unsafe plan approach",
        )

    config = AssistantConfig(auto_approve_review=False)
    assistant = PlanAndExecuteAssistant(
        llm_service=mock_llm_service,
        executor=mock_executor,
        memory_manager=mock_mem,
        store=mock_store,
        plan_generator=mock_generator,
        config=config,
    )

    result = await assistant.run(
        goal="Unsafe goal",
        user_id="user_test",
        decision_provider=mock_decision_provider,
    )

    assert result.status == RunStatus.CANCELLED
    assert "rejected" in str(result.error_message).lower()


@pytest.mark.asyncio
async def test_assistant_dynamic_replanning_recovery(
    mock_llm_service: MagicMock,
    sample_valid_plan: Plan,
) -> None:
    """Verify failing step triggers dynamic replanning and recovers to COMPLETED."""
    mock_executor = MagicMock()
    mock_executor.get_available_capabilities = MagicMock(return_value={})

    mock_mem = MagicMock()
    mock_mem.get_user_preferences = AsyncMock(return_value=[])
    mock_mem.record_plan_episode = AsyncMock()

    mock_store = MagicMock()
    mock_store.save_run = AsyncMock()
    mock_store._save_run_sync = MagicMock()

    mock_generator = MagicMock()
    mock_generator.generate_plan = AsyncMock(return_value=sample_valid_plan)

    # First schedule attempt fails on step_2; second attempt succeeds
    mock_scheduler = MagicMock()
    mock_scheduler.execute_plan = AsyncMock(
        side_effect=[
            SchedulerExecutionResult(
                status=StepStatus.FAILED,
                failed_step_id="step_2",
                error_message="API connection refused",
                step_results={"step_1": "Verified C-101"},
            ),
            SchedulerExecutionResult(
                status=StepStatus.COMPLETED,
                step_results={"step_1": "Verified C-101", "step_2_alt": "Credit via DB"},
            ),
        ]
    )

    revised_plan = Plan(
        goal=sample_valid_plan.goal,
        version=2,
        parent_version=1,
        created_by="replanner",
        steps=[
            PlanStep(
                id="step_1",
                description="Fetch",
                success_criteria="Done",
                status=StepStatus.COMPLETED,
            ),
            PlanStep(
                id="step_2_alt",
                description="Credit via DB",
                depends_on=["step_1"],
                success_criteria="Done",
            ),
        ],
    )

    mock_replanner = MagicMock()
    mock_replanner.replan = AsyncMock(
        return_value=ReplanResult(
            new_plan=revised_plan,
            replan_count=1,
            preserved_step_ids=["step_1"],
            new_step_ids=["step_2_alt"],
        )
    )

    config = AssistantConfig(auto_approve_review=True)
    assistant = PlanAndExecuteAssistant(
        llm_service=mock_llm_service,
        executor=mock_executor,
        memory_manager=mock_mem,
        store=mock_store,
        plan_generator=mock_generator,
        scheduler=mock_scheduler,
        replanner=mock_replanner,
        config=config,
    )

    result = await assistant.run(
        goal="Investigate and recover",
        user_id="user_test",
    )

    assert result.status == RunStatus.COMPLETED
    assert result.plan.version == 2
    mock_replanner.replan.assert_awaited_once()


@pytest.mark.asyncio
async def test_assistant_budget_exceeded_halt(
    mock_llm_service: MagicMock,
    sample_valid_plan: Plan,
) -> None:
    """Verify assistant halts with status FAILED when budget is exhausted."""
    mock_executor = MagicMock()
    mock_executor.get_available_capabilities = MagicMock(return_value={})

    mock_mem = MagicMock()
    mock_mem.get_user_preferences = AsyncMock(return_value=[])

    mock_store = MagicMock()
    mock_store.save_run = AsyncMock()
    mock_store._save_run_sync = MagicMock()

    mock_generator = MagicMock()
    mock_generator.generate_plan = AsyncMock(return_value=sample_valid_plan)

    config = AssistantConfig(auto_approve_review=True)
    assistant = PlanAndExecuteAssistant(
        llm_service=mock_llm_service,
        executor=mock_executor,
        memory_manager=mock_mem,
        store=mock_store,
        plan_generator=mock_generator,
        config=config,
    )

    # Budget already exhausted
    tight_budget = RunBudget(max_total_tokens=100)
    tight_budget.start_timer()
    tight_budget.record_usage(tokens=101, cost_usd=0.0)

    result = await assistant.run(
        goal="Budget constrained run",
        user_id="user_test",
        budget=tight_budget,
    )

    assert result.status == RunStatus.FAILED
    assert "budget exceeded" in str(result.error_message).lower()


@pytest.mark.asyncio
async def test_assistant_optional_reflector_integration(
    mock_llm_service: MagicMock,
    sample_valid_plan: Plan,
) -> None:
    """Verify optional PlanReflector hook is invoked for Module 19 self-reflection."""
    mock_executor = MagicMock()
    mock_executor.get_available_capabilities = MagicMock(return_value={})

    mock_mem = MagicMock()
    mock_mem.get_user_preferences = AsyncMock(return_value=[])
    mock_mem.record_plan_episode = AsyncMock()

    mock_store = MagicMock()
    mock_store.save_run = AsyncMock()
    mock_store._save_run_sync = MagicMock()

    mock_generator = MagicMock()
    mock_generator.generate_plan = AsyncMock(return_value=sample_valid_plan)

    mock_scheduler = MagicMock()
    mock_scheduler.execute_plan = AsyncMock(
        return_value=SchedulerExecutionResult(
            status=StepStatus.COMPLETED,
            step_results={"step_1": "Done", "step_2": "Done"},
        )
    )

    mock_reflector = MagicMock()
    mock_reflector.reflect = AsyncMock(
        return_value=ReflectionResult(
            should_replan=False,
            critique="High quality execution plan",
            score=0.98,
        )
    )

    config = AssistantConfig(auto_approve_review=True)
    assistant = PlanAndExecuteAssistant(
        llm_service=mock_llm_service,
        executor=mock_executor,
        memory_manager=mock_mem,
        store=mock_store,
        plan_generator=mock_generator,
        scheduler=mock_scheduler,
        reflector=mock_reflector,
        config=config,
    )

    result = await assistant.run(
        goal="Test with reflection",
        user_id="user_reflect",
    )

    assert result.status == RunStatus.COMPLETED
    mock_reflector.reflect.assert_awaited_once()


@pytest.mark.asyncio
async def test_assistant_episodic_memory_failure_does_not_fail_run(
    mock_llm_service: MagicMock,
    sample_valid_plan: Plan,
) -> None:
    """Verify that transient episodic memory failures do not cause a successful run to fail."""
    mock_executor = MagicMock()
    mock_executor.get_available_capabilities = MagicMock(return_value={"tool_a": "Tool A"})

    mock_mem = MagicMock()
    mock_mem.get_user_preferences = AsyncMock(return_value=[])
    # Memory record throws database connection error
    mock_mem.record_plan_episode = AsyncMock(
        side_effect=RuntimeError("PostgreSQL connection lost during episodic insert")
    )

    mock_store = MagicMock()
    mock_store.save_run = AsyncMock()
    mock_store._save_run_sync = MagicMock()

    mock_generator = MagicMock()
    mock_generator.generate_plan = AsyncMock(return_value=sample_valid_plan)

    mock_scheduler = MagicMock()
    mock_scheduler.execute_plan = AsyncMock(
        return_value=SchedulerExecutionResult(
            status=StepStatus.COMPLETED,
            step_results={"step_1": "Verified step 1", "step_2": "Verified step 2"},
        )
    )

    config = AssistantConfig(auto_approve_review=True)
    assistant = PlanAndExecuteAssistant(
        llm_service=mock_llm_service,
        executor=mock_executor,
        memory_manager=mock_mem,
        store=mock_store,
        plan_generator=mock_generator,
        scheduler=mock_scheduler,
        config=config,
    )

    result = await assistant.run(
        goal="Test resilience to episodic memory logging failure",
        user_id="user_resilient",
    )

    # Run MUST complete successfully despite episodic memory write failure!
    assert result.status == RunStatus.COMPLETED
    assert result.final_deliverable is not None
    mock_mem.record_plan_episode.assert_awaited_once()


@pytest.mark.asyncio
async def test_assistant_finalizes_run_with_pure_async_store(
    mock_llm_service: MagicMock,
    sample_valid_plan: Plan,
) -> None:
    """Verify assistant finalizes run and persists RunRecord using pure async save_run (PLN-008)."""
    from unittest.mock import create_autospec

    from plan_and_execute_research_assistant.storage.protocol import RunRecordStore

    # Custom store implementing only the RunRecordStore async protocol (no _save_run_sync attribute)
    pure_async_store = create_autospec(RunRecordStore, instance=True)
    pure_async_store.save_run = AsyncMock()

    mock_executor = MagicMock()
    mock_executor.get_available_capabilities = MagicMock(return_value={"tool_a": "Tool A"})

    mock_mem = MagicMock()
    mock_mem.get_user_preferences = AsyncMock(return_value=[])
    mock_mem.record_plan_episode = AsyncMock()

    mock_generator = MagicMock()
    mock_generator.generate_plan = AsyncMock(return_value=sample_valid_plan)

    mock_scheduler = MagicMock()
    mock_scheduler.execute_plan = AsyncMock(
        return_value=SchedulerExecutionResult(
            status=StepStatus.COMPLETED,
            step_results={"step_1": "Result 1", "step_2": "Result 2"},
        )
    )

    assistant = PlanAndExecuteAssistant(
        llm_service=mock_llm_service,
        executor=mock_executor,
        memory_manager=mock_mem,
        store=pure_async_store,
        plan_generator=mock_generator,
        scheduler=mock_scheduler,
        config=AssistantConfig(auto_approve_review=True),
    )

    result = await assistant.run(goal="Test async finalize", user_id="user_async")

    assert result.status == RunStatus.COMPLETED
    assert not hasattr(pure_async_store, "_save_run_sync")
    # Verify save_run was called asynchronously
    assert pure_async_store.save_run.await_count >= 1


@pytest.mark.asyncio
async def test_assistant_preserves_immutable_initial_plan_in_plan_versions(
    mock_llm_service: MagicMock,
    sample_valid_plan: Plan,
) -> None:
    """Verify that run_record.plan_versions[0] preserves pre-execution pending status (PLN-014)."""
    mock_executor = MagicMock()
    mock_executor.get_available_capabilities = MagicMock(return_value={"tool_a": "Tool A"})

    mock_mem = MagicMock()
    mock_mem.get_user_preferences = AsyncMock(return_value=[])
    mock_mem.record_plan_episode = AsyncMock()

    mock_store = MagicMock()
    mock_store.save_run = AsyncMock()

    mock_generator = MagicMock()
    # sample_valid_plan has steps with status=PENDING
    assert sample_valid_plan.steps[0].status == StepStatus.PENDING
    mock_generator.generate_plan = AsyncMock(return_value=sample_valid_plan)

    async def fake_scheduler_run(plan: Plan, **kwargs: Any) -> SchedulerExecutionResult:
        # Scheduler mutates executed plan's steps in-place
        for s in plan.steps:
            s.status = StepStatus.COMPLETED
            s.result_summary = f"Completed {s.id}"
        return SchedulerExecutionResult(
            status=StepStatus.COMPLETED,
            step_results={s.id: s.result_summary or "" for s in plan.steps},
        )

    mock_scheduler = MagicMock()
    mock_scheduler.execute_plan.side_effect = fake_scheduler_run

    assistant = PlanAndExecuteAssistant(
        llm_service=mock_llm_service,
        executor=mock_executor,
        memory_manager=mock_mem,
        store=mock_store,
        plan_generator=mock_generator,
        scheduler=mock_scheduler,
        config=AssistantConfig(auto_approve_review=True),
    )

    result = await assistant.run(goal="Test plan_versions immutability", user_id="user_immut")

    assert result.status == RunStatus.COMPLETED
    # The active/returned plan has completed steps
    assert result.plan.steps[0].status == StepStatus.COMPLETED
    # But plan_versions[0] MUST retain its pristine pre-execution snapshot (PENDING)
    assert result.run_record.plan_versions[0].steps[0].status == StepStatus.PENDING
