"""Unit tests for DAGScheduler, bounded concurrency, and sibling settling."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from plan_and_execute_research_assistant.core import AssistantConfig
from plan_and_execute_research_assistant.execution import Module17StepExecutor, StepExecutionOutput
from plan_and_execute_research_assistant.models import (
    Plan,
    PlanStep,
    RunBudget,
    StepStatus,
    StepVerificationResult,
)
from plan_and_execute_research_assistant.scheduling import (
    DAGDeadlockError,
    DAGScheduler,
)
from plan_and_execute_research_assistant.verification import StepVerifier


@pytest.fixture
def mock_verifier() -> MagicMock:
    """Mock StepVerifier that passes by default."""
    verifier = MagicMock(spec=StepVerifier)
    verifier.verify_step = AsyncMock(
        return_value=StepVerificationResult(
            step_id="test",
            passed=True,
            programmatic_passed=True,
            model_passed=True,
            score=0.95,
            reasoning="Valid deliverable",
        )
    )
    return verifier


@pytest.fixture
def mock_executor() -> MagicMock:
    """Mock StepExecutor."""
    executor = MagicMock(spec=Module17StepExecutor)
    executor.execute_step = AsyncMock()
    return executor


@pytest.mark.asyncio
async def test_scheduler_linear_pipeline(
    mock_executor: MagicMock, mock_verifier: MagicMock
) -> None:
    """Verify linear dependency chain executes in order and propagates predecessor outputs."""
    plan = Plan(
        goal="Linear pipeline test",
        steps=[
            PlanStep(id="s1", description="Extract data", success_criteria="Done"),
            PlanStep(
                id="s2", description="Transform data", depends_on=["s1"], success_criteria="Done"
            ),
            PlanStep(id="s3", description="Load data", depends_on=["s2"], success_criteria="Done"),
        ],
    )

    execution_order: list[str] = []

    async def fake_execute(
        step: PlanStep, predecessor_context: dict[str, str], **kwargs
    ) -> StepExecutionOutput:
        execution_order.append(step.id)
        if step.id == "s2":
            assert "s1" in predecessor_context
            assert predecessor_context["s1"] == "Output of s1"
        if step.id == "s3":
            assert "s2" in predecessor_context
            assert predecessor_context["s2"] == "Output of s2"

        return StepExecutionOutput(
            step_id=step.id,
            output_text=f"Output of {step.id}",
            status=StepStatus.COMPLETED,
            tokens_used=100,
            cost_usd=0.001,
        )

    mock_executor.execute_step.side_effect = fake_execute

    scheduler = DAGScheduler(executor=mock_executor, verifier=mock_verifier)
    result = await scheduler.execute_plan(plan=plan, user_id="user_test")

    assert result.status == StepStatus.COMPLETED
    assert execution_order == ["s1", "s2", "s3"]
    assert len(result.step_results) == 3
    assert result.step_results["s1"] == "Output of s1"


@pytest.mark.asyncio
async def test_scheduler_parallel_diamond_dag(
    mock_executor: MagicMock, mock_verifier: MagicMock
) -> None:
    """Verify diamond DAG executes intermediate sibling steps concurrently."""
    plan = Plan(
        goal="Diamond DAG test",
        steps=[
            PlanStep(id="root", description="Fetch raw source", success_criteria="Done"),
            PlanStep(
                id="branch_a",
                description="Process part A",
                depends_on=["root"],
                success_criteria="Done",
            ),
            PlanStep(
                id="branch_b",
                description="Process part B",
                depends_on=["root"],
                success_criteria="Done",
            ),
            PlanStep(
                id="join",
                description="Join A and B",
                depends_on=["branch_a", "branch_b"],
                success_criteria="Done",
            ),
        ],
    )

    concurrent_branches = 0
    max_concurrent_seen = 0

    async def fake_execute(
        step: PlanStep, predecessor_context: dict[str, str], **kwargs
    ) -> StepExecutionOutput:
        nonlocal concurrent_branches, max_concurrent_seen
        if "branch" in step.id:
            concurrent_branches += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_branches)
            await asyncio.sleep(0.05)  # simulate async work
            concurrent_branches -= 1

        if step.id == "join":
            assert "branch_a" in predecessor_context
            assert "branch_b" in predecessor_context

        return StepExecutionOutput(
            step_id=step.id,
            output_text=f"Output of {step.id}",
            status=StepStatus.COMPLETED,
            tokens_used=50,
            cost_usd=0.0005,
        )

    mock_executor.execute_step.side_effect = fake_execute

    scheduler = DAGScheduler(executor=mock_executor, verifier=mock_verifier)
    result = await scheduler.execute_plan(plan=plan, user_id="user_test")

    assert result.status == StepStatus.COMPLETED
    assert max_concurrent_seen == 2
    assert "join" in result.step_results


@pytest.mark.asyncio
async def test_scheduler_enforces_concurrency_cap(
    mock_executor: MagicMock, mock_verifier: MagicMock
) -> None:
    """Verify semaphore strictly bounds concurrent tasks to max_concurrency."""
    # 6 independent root steps
    plan = Plan(
        goal="Concurrency cap test",
        steps=[
            PlanStep(id=f"step_{i}", description=f"Task {i}", success_criteria="Done")
            for i in range(6)
        ],
    )

    current_running = 0
    peak_concurrency = 0
    lock = asyncio.Lock()

    async def fake_execute(step: PlanStep, **kwargs) -> StepExecutionOutput:
        nonlocal current_running, peak_concurrency
        async with lock:
            current_running += 1
            peak_concurrency = max(peak_concurrency, current_running)

        await asyncio.sleep(0.04)

        async with lock:
            current_running -= 1

        return StepExecutionOutput(
            step_id=step.id,
            output_text=f"Done {step.id}",
            status=StepStatus.COMPLETED,
        )

    mock_executor.execute_step.side_effect = fake_execute

    config = AssistantConfig(max_concurrency=2)
    scheduler = DAGScheduler(executor=mock_executor, verifier=mock_verifier, config=config)
    result = await scheduler.execute_plan(plan=plan, user_id="user_test")

    assert result.status == StepStatus.COMPLETED
    assert peak_concurrency == 2  # capped strictly at 2
    assert len(result.step_results) == 6


@pytest.mark.asyncio
async def test_scheduler_step_retry_with_feedback_success(
    mock_executor: MagicMock, mock_verifier: MagicMock
) -> None:
    """Verify failed verification triggers local retry with diagnostic feedback."""
    plan = Plan(
        goal="Retry test",
        steps=[PlanStep(id="s1", description="Produce metrics", success_criteria="Full metrics")],
    )

    attempt_count = 0
    feedback_received: list[str | None] = []

    async def fake_execute(
        step: PlanStep, retry_feedback: str | None = None, **kwargs
    ) -> StepExecutionOutput:
        nonlocal attempt_count
        attempt_count += 1
        feedback_received.append(retry_feedback)
        return StepExecutionOutput(
            step_id=step.id,
            output_text=f"Attempt {attempt_count} output",
            status=StepStatus.COMPLETED,
        )

    mock_executor.execute_step.side_effect = fake_execute

    # Attempt 1 fails, Attempt 2 passes
    mock_verifier.verify_step.side_effect = [
        StepVerificationResult(
            step_id="s1",
            passed=False,
            programmatic_passed=True,
            model_passed=False,
            score=0.60,
            reasoning="Missing latency metric",
            missing_criteria=["Latency in ms"],
        ),
        StepVerificationResult(
            step_id="s1",
            passed=True,
            programmatic_passed=True,
            model_passed=True,
            score=0.95,
            reasoning="All metrics present",
        ),
    ]

    config = AssistantConfig(max_step_retries=2)
    scheduler = DAGScheduler(executor=mock_executor, verifier=mock_verifier, config=config)
    result = await scheduler.execute_plan(plan=plan, user_id="user_test")

    assert result.status == StepStatus.COMPLETED
    assert attempt_count == 2
    assert feedback_received[0] is None
    assert feedback_received[1] is not None
    assert "Missing latency metric" in feedback_received[1]
    assert len(result.step_records["s1"]) == 2


@pytest.mark.asyncio
async def test_scheduler_sibling_settling_on_failure(
    mock_executor: MagicMock, mock_verifier: MagicMock
) -> None:
    """Verify concurrency-safe sibling settling: Step B settles cleanly when Step A fails."""
    plan = Plan(
        goal="Sibling settling test",
        steps=[
            PlanStep(id="step_a", description="Failing task", success_criteria="Done"),
            PlanStep(id="step_b", description="Slow successful sibling", success_criteria="Done"),
        ],
    )

    async def fake_execute(step: PlanStep, **kwargs) -> StepExecutionOutput:
        if step.id == "step_a":
            # Immediately fails
            return StepExecutionOutput(
                step_id="step_a",
                output_text="Bad output",
                status=StepStatus.FAILED,
            )
        # step_b takes slightly longer but completes
        await asyncio.sleep(0.05)
        return StepExecutionOutput(
            step_id="step_b",
            output_text="Valid sibling result",
            status=StepStatus.COMPLETED,
        )

    mock_executor.execute_step.side_effect = fake_execute

    # step_a fails verification on all attempts; step_b passes
    async def fake_verify(step: PlanStep, output_text: str, **kwargs) -> StepVerificationResult:
        if step.id == "step_a":
            return StepVerificationResult(
                step_id="step_a",
                passed=False,
                programmatic_passed=False,
                model_passed=False,
                score=0.0,
                reasoning="Step A failed execution",
            )
        return StepVerificationResult(
            step_id="step_b",
            passed=True,
            programmatic_passed=True,
            model_passed=True,
            score=0.95,
            reasoning="Step B verified",
        )

    mock_verifier.verify_step.side_effect = fake_verify

    config = AssistantConfig(max_step_retries=0)  # no retries for fast test
    scheduler = DAGScheduler(executor=mock_executor, verifier=mock_verifier, config=config)
    result = await scheduler.execute_plan(plan=plan, user_id="user_test")

    # Overall run status is FAILED
    assert result.status == StepStatus.FAILED
    assert result.failed_step_id == "step_a"

    # CRITICAL: Step B completed and its result was preserved!
    assert "step_b" in result.step_results
    assert result.step_results["step_b"] == "Valid sibling result"


@pytest.mark.asyncio
async def test_scheduler_deadlock_detection(
    mock_executor: MagicMock, mock_verifier: MagicMock
) -> None:
    """Verify DAGDeadlockError is raised if unresolvable dependency cycle or deadlock exists."""
    # Step has an impossible dependency on an undeclared step or cycle
    step_a = PlanStep(
        id="step_a", description="Task A", depends_on=["unmet_step"], success_criteria="Done"
    )
    plan = Plan(goal="Deadlock plan", steps=[step_a])

    scheduler = DAGScheduler(executor=mock_executor, verifier=mock_verifier)
    with pytest.raises(DAGDeadlockError, match="deadlock detected"):
        await scheduler.execute_plan(plan=plan, user_id="user_test")


@pytest.mark.asyncio
async def test_scheduler_budget_recording(
    mock_executor: MagicMock, mock_verifier: MagicMock
) -> None:
    """Verify token and cost consumption are tracked inside RunBudget."""
    plan = Plan(
        goal="Budget tracking test",
        steps=[PlanStep(id="s1", description="Task 1", success_criteria="Done")],
    )

    mock_executor.execute_step.return_value = StepExecutionOutput(
        step_id="s1",
        output_text="Valid output satisfying criteria length requirement.",
        status=StepStatus.COMPLETED,
        tokens_used=2500,
        cost_usd=0.015,
    )

    budget = RunBudget(max_total_tokens=10_000, max_total_cost_usd=0.10)
    scheduler = DAGScheduler(executor=mock_executor, verifier=mock_verifier)
    result = await scheduler.execute_plan(plan=plan, user_id="user_test", budget=budget)

    assert result.status == StepStatus.COMPLETED
    assert budget.current_tokens == 2500
    assert budget.current_cost_usd == 0.015


@pytest.mark.asyncio
async def test_scheduler_step_records_collector_preserves_executions_on_budget_exceeded(
    mock_executor: MagicMock, mock_verifier: MagicMock
) -> None:
    from plan_and_execute_research_assistant.models import BudgetExceededError
    from plan_and_execute_research_assistant.models.run_record import StepExecutionRecord

    plan = Plan(
        goal="Budget exceeded test",
        steps=[
            PlanStep(id="s1", description="Task 1", success_criteria="Done"),
            PlanStep(id="s2", description="Task 2", depends_on=["s1"], success_criteria="Done"),
        ],
    )

    mock_executor.execute_step.return_value = StepExecutionOutput(
        step_id="s1",
        output_text="Output that exceeds budget immediately.",
        status=StepStatus.COMPLETED,
        tokens_used=5000,
        cost_usd=0.05,
    )

    # Budget cap is 1,000 tokens; step uses 5,000 tokens so next budget check / record will raise BudgetExceededError
    budget = RunBudget(max_total_tokens=1000, max_total_cost_usd=0.10)
    collector: dict[str, list[StepExecutionRecord]] = {}

    scheduler = DAGScheduler(executor=mock_executor, verifier=mock_verifier)

    with pytest.raises(BudgetExceededError, match="Token budget exhausted"):
        await scheduler.execute_plan(
            plan=plan,
            user_id="user_test",
            budget=budget,
            step_records_collector=collector,
        )

    # Telemetry was preserved in real time despite the exception!
    assert "s1" in collector
    assert len(collector["s1"]) == 1
    assert collector["s1"][0].step_id == "s1"
    assert collector["s1"][0].tokens_used == 5000
