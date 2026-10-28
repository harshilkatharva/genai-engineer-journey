"""Asynchronous Directed Acyclic Graph (DAG) scheduler."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from genai_core import ObservabilityEvent, safe_log

from ..core import AssistantConfig, get_assistant_config
from ..models import (
    BudgetExceededError,
    Plan,
    PlanStep,
    RunBudget,
    StepExecutionRecord,
    StepStatus,
    StepVerificationResult,
)
from ..protocols.specialist import StepExecutor
from ..verification.verifier import StepVerifier
from .exceptions import DAGDeadlockError

_logger = logging.getLogger("plan_and_execute_research_assistant.scheduling")


@dataclass
class SchedulerExecutionResult:
    """Consolidated outcome of DAGScheduler plan execution."""

    status: StepStatus
    step_results: dict[str, str] = field(default_factory=dict)
    step_records: dict[str, list[StepExecutionRecord]] = field(default_factory=dict)
    failed_step_id: str | None = None
    error_message: str | None = None


class DAGScheduler:
    """Pure-Python asynchronous DAG scheduler with bounded concurrency and sibling settling.

    Dispatches independent steps concurrently bounded by an asyncio.Semaphore.
    Propagates verified predecessor outputs into dependent step prompts.
    Retries failing steps locally up to max_step_retries before marking failure.
    Settles running sibling tasks safely upon any step failure.
    """

    def __init__(
        self,
        executor: StepExecutor,
        verifier: StepVerifier,
        config: AssistantConfig | None = None,
    ) -> None:
        """Initialize scheduler with executor, dual-gate verifier, and configuration."""
        self.executor = executor
        self.verifier = verifier
        self.config = config or get_assistant_config()

    async def execute_plan(
        self,
        plan: Plan,
        user_id: str,
        budget: RunBudget | None = None,
        step_records_collector: dict[str, list[StepExecutionRecord]] | None = None,
    ) -> SchedulerExecutionResult:
        """Execute all steps in the plan DAG respecting dependencies and concurrency limits.

        Args:
            plan: Validated Plan DAG to execute.
            user_id: User identifier.
            budget: Optional RunBudget for token/cost/timeout enforcement.
            step_records_collector: Optional live collector dictionary to stream StepExecutionRecord
                instances immediately as each attempt completes (prevents telemetry loss on budget exhaustion).

        Returns:
            SchedulerExecutionResult detailing final status, output summaries, and telemetry.
        """
        active_budget = budget or RunBudget(
            max_concurrency=self.config.max_concurrency,
            max_step_retries=self.config.max_step_retries,
        )
        active_budget.start_timer()

        # In-memory execution state (preserving previously completed steps from replans)
        step_status: dict[str, StepStatus] = {
            s.id: s.status if s.status == StepStatus.COMPLETED else StepStatus.PENDING
            for s in plan.steps
        }
        step_results: dict[str, str] = {
            s.id: s.result_summary
            for s in plan.steps
            if s.status == StepStatus.COMPLETED and s.result_summary
        }
        step_records: dict[str, list[StepExecutionRecord]] = {s.id: [] for s in plan.steps}

        active_tasks: dict[str, asyncio.Task[None]] = {}
        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        failed_step_id: str | None = None
        error_message: str | None = None

        safe_log(
            _logger,
            logging.INFO,
            f"DAGScheduler starting execution of plan version {plan.version} ({len(plan.steps)} steps, concurrency cap: {self.config.max_concurrency})",
            event=ObservabilityEvent.TOOL_LOOP_START,
            plan_version=plan.version,
            total_steps=len(plan.steps),
            max_concurrency=self.config.max_concurrency,
        )

        async def _run_single_step(step: PlanStep) -> None:
            nonlocal failed_step_id, error_message

            async with semaphore:
                step_status[step.id] = StepStatus.RUNNING
                step.status = StepStatus.RUNNING

                # Extract predecessor context from verified dependency outputs
                predecessor_context: dict[str, str] = {}
                for dep_id in step.depends_on or []:
                    if dep_id in step_results:
                        predecessor_context[dep_id] = step_results[dep_id]

                retry_feedback: str | None = None
                attempt = 0
                max_retries = self.config.max_step_retries

                while attempt <= max_retries:
                    attempt += 1
                    step.attempts = attempt

                    # Enforce global budget before dispatching attempt
                    active_budget.check_budget()

                    attempt_start = datetime.now(UTC)
                    try:
                        exec_output = await self.executor.execute_step(
                            step=step,
                            predecessor_context=predecessor_context,
                            user_id=user_id,
                            retry_feedback=retry_feedback,
                        )
                    except Exception as exc:  # noqa: BLE001
                        safe_log(
                            _logger,
                            logging.ERROR,
                            f"Step '{step.id}' attempt {attempt} executor threw error: {exc}",
                            step_id=step.id,
                            attempt=attempt,
                            error=str(exc),
                        )
                        from ..execution import StepExecutionOutput

                        exec_output = StepExecutionOutput(
                            step_id=step.id,
                            output_text="",
                            status=StepStatus.FAILED,
                            error_message=str(exc),
                        )

                    # Track token & cost usage in RunBudget
                    tokens = getattr(exec_output, "tokens_used", 0)
                    cost = getattr(exec_output, "cost_usd", 0.0)
                    active_budget.record_usage(tokens=tokens, cost_usd=cost)

                    output_text = getattr(exec_output, "output_text", "")
                    exec_status = getattr(exec_output, "status", StepStatus.FAILED)
                    dur_ms = getattr(exec_output, "duration_ms", 0.0)

                    # Verify deliverable through dual-gate verifier
                    verification: StepVerificationResult = await self.verifier.verify_step(
                        step=step,
                        output_text=output_text,
                        execution_status=exec_status,
                    )

                    # Track token & cost usage from model verification in RunBudget
                    v_tokens = getattr(verification, "tokens_used", 0) or 0
                    v_cost = getattr(verification, "cost_usd", 0.0) or 0.0
                    if v_tokens or v_cost:
                        active_budget.record_usage(tokens=v_tokens, cost_usd=v_cost)

                    # Total attempt tokens & cost (executor + verifier)
                    attempt_tokens = tokens + v_tokens
                    attempt_cost = cost + v_cost

                    # Record attempt telemetry
                    record = StepExecutionRecord(
                        step_id=step.id,
                        plan_version=plan.version,
                        status=StepStatus.COMPLETED if verification.passed else StepStatus.FAILED,
                        attempts=attempt,
                        depends_on=list(step.depends_on or []),
                        started_at=attempt_start,
                        completed_at=datetime.now(UTC),
                        tokens_used=attempt_tokens,
                        cost_usd=attempt_cost,
                        duration_ms=dur_ms,
                        result_summary=output_text if verification.passed else None,
                        verification=verification,
                        error_message=verification.reasoning if not verification.passed else None,
                    )
                    step_records[step.id].append(record)
                    if step_records_collector is not None:
                        if step.id not in step_records_collector:
                            step_records_collector[step.id] = []
                        step_records_collector[step.id].append(record)

                    if verification.passed:
                        step_status[step.id] = StepStatus.COMPLETED
                        step.status = StepStatus.COMPLETED
                        step.result_summary = output_text
                        step_results[step.id] = output_text

                        safe_log(
                            _logger,
                            logging.INFO,
                            f"Step '{step.id}' completed and verified on attempt {attempt}/{max_retries + 1}",
                            step_id=step.id,
                            attempt=attempt,
                        )
                        return

                    # Verification failed: format feedback for next retry
                    safe_log(
                        _logger,
                        logging.WARNING,
                        f"Step '{step.id}' attempt {attempt} failed verification: {verification.reasoning}",
                        step_id=step.id,
                        attempt=attempt,
                        score=verification.score,
                    )
                    missing_str = (
                        ", ".join(verification.missing_criteria)
                        if verification.missing_criteria
                        else "None"
                    )
                    retry_feedback = (
                        f"Reasoning: {verification.reasoning}\nMissing Criteria: {missing_str}"
                    )

                # All retries exhausted
                step_status[step.id] = StepStatus.FAILED
                step.status = StepStatus.FAILED
                step.error_message = (
                    f"Failed verification after {attempt} attempts: {retry_feedback}"
                )
                failed_step_id = step.id
                error_message = step.error_message

                safe_log(
                    _logger,
                    logging.ERROR,
                    f"Step '{step.id}' exhausted retries without passing verification.",
                    step_id=step.id,
                    attempts=attempt,
                )

        # Main orchestration loop
        try:
            while True:
                # 1. Clean up completed tasks from active_tasks map and inspect exceptions
                done_task_ids = [s_id for s_id, t in active_tasks.items() if t.done()]
                for s_id in done_task_ids:
                    t = active_tasks.pop(s_id)
                    exc = t.exception()
                    if exc is not None:
                        if isinstance(exc, BudgetExceededError):
                            for active_t in active_tasks.values():
                                if not active_t.done():
                                    active_t.cancel()
                            raise exc
                        step_status[s_id] = StepStatus.FAILED
                        step_obj = next((s for s in plan.steps if s.id == s_id), None)
                        if step_obj:
                            step_obj.status = StepStatus.FAILED
                            step_obj.error_message = str(exc)
                        if failed_step_id is None:
                            failed_step_id = s_id
                            error_message = str(exc)

                # 2. Check for failure condition
                if failed_step_id is not None or any(
                    st == StepStatus.FAILED for st in step_status.values()
                ):
                    safe_log(
                        _logger,
                        logging.WARNING,
                        f"Step failure detected ({failed_step_id}). Pausing new dispatch and settling {len(active_tasks)} running sibling tasks...",
                        failed_step_id=failed_step_id,
                        running_siblings=len(active_tasks),
                    )
                    # Sibling Settling: await active sibling tasks to complete without aborting
                    if active_tasks:
                        await asyncio.gather(*active_tasks.values(), return_exceptions=True)

                    return SchedulerExecutionResult(
                        status=StepStatus.FAILED,
                        step_results=step_results,
                        step_records=step_records,
                        failed_step_id=failed_step_id,
                        error_message=error_message or f"Step '{failed_step_id}' failed execution.",
                    )

                # 3. Check for overall plan completion
                if all(st == StepStatus.COMPLETED for st in step_status.values()):
                    safe_log(
                        _logger,
                        logging.INFO,
                        f"All {len(plan.steps)} steps in plan version {plan.version} successfully completed and verified.",
                        plan_version=plan.version,
                    )
                    return SchedulerExecutionResult(
                        status=StepStatus.COMPLETED,
                        step_results=step_results,
                        step_records=step_records,
                    )

                # 4. Resolve and dispatch READY steps
                ready_steps: list[PlanStep] = []
                for s in plan.steps:
                    if step_status[s.id] != StepStatus.PENDING or s.id in active_tasks:
                        continue

                    # Check if all dependencies are COMPLETED
                    deps = s.depends_on or []
                    if all(step_status.get(d) == StepStatus.COMPLETED for d in deps):
                        ready_steps.append(s)

                if ready_steps:
                    for s in ready_steps:
                        step_status[s.id] = StepStatus.READY
                        s.status = StepStatus.READY
                        task = asyncio.create_task(_run_single_step(s))
                        active_tasks[s.id] = task

                # 5. Deadlock detection: no ready steps and no active tasks
                if not active_tasks:
                    unresolved = [
                        s
                        for s, st in step_status.items()
                        if st not in (StepStatus.COMPLETED, StepStatus.FAILED)
                    ]
                    raise DAGDeadlockError(
                        f"DAG scheduler deadlock detected. Unresolved steps: {unresolved}. "
                        "Dependencies could not be satisfied."
                    )

                # 6. Await at least one active task to complete
                await asyncio.wait(active_tasks.values(), return_when=asyncio.FIRST_COMPLETED)

        except Exception:
            # Settle running tasks on unexpected cancellations or errors
            if active_tasks:
                for t in active_tasks.values():
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*active_tasks.values(), return_exceptions=True)
            raise
