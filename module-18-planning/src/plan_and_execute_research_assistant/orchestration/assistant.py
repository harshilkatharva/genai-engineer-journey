"""Plan-and-Execute Research Assistant facade orchestrator."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from genai_core import MemoryService, ObservabilityEvent, safe_log
from pydantic import BaseModel, Field

from ..core import AssistantConfig, get_assistant_config
from ..execution import Module17StepExecutor
from ..memory import PlanningMemoryManager
from ..models import (
    BudgetExceededError,
    Plan,
    PlanReviewDecision,
    PlanReviewRequest,
    RunBudget,
    RunRecord,
    RunStatus,
    StepStatus,
)
from ..planning import PlanGenerator
from ..protocols import PlanReflector, StepExecutor
from ..replanning import ReplanningEngine
from ..review import HumanPlanReviewGateway, PlanRejectedError
from ..scheduling import DAGScheduler
from ..storage import FileRunRecordStore, RunRecordStore
from ..verification import StepVerifier
from .prompts import build_synthesis_prompt

if TYPE_CHECKING:
    from genai_core import LLMService

DecisionProvider = Callable[[PlanReviewRequest], Awaitable[PlanReviewDecision] | PlanReviewDecision]

_logger = logging.getLogger("plan_and_execute_research_assistant.orchestration")


class AssistantRunResult(BaseModel):
    """Consolidated outcome returned by PlanAndExecuteAssistant."""

    run_id: str = Field(description="Unique execution run identifier")
    status: RunStatus = Field(description="Final execution lifecycle status")
    final_deliverable: str | None = Field(
        default=None, description="Synthesized final deliverable text"
    )
    plan: Plan = Field(description="Final Plan DAG associated with the run")
    run_record: RunRecord = Field(description="Full audit record containing all step traces")
    total_tokens: int = Field(default=0, ge=0, description="Cumulative tokens consumed")
    total_cost_usd: float = Field(
        default=0.0, ge=0.0, description="Cumulative cost incurred in USD"
    )
    duration_ms: float = Field(
        default=0.0, ge=0.0, description="Total wall-clock execution time in ms"
    )
    error_message: str | None = Field(default=None, description="Error detail if execution failed")


class PlanAndExecuteAssistant:
    """Production-grade orchestrator facade for the Plan-and-Execute Research Assistant.

    Coordinates the 5-phase lifecycle:
    - Phase 1: Context & memory retrieval (LTM preferences + episodic past plans)
    - Phase 2: Structured plan generation & automated repair loop
    - Phase 3: Human plan review gating (approval, modification, or rejection)
    - Phase 4: Bounded concurrent DAG execution and concurrency-safe replanning
    - Phase 5: Synthesis of final verified deliverable and atomic run record persistence
    """

    def __init__(
        self,
        llm_service: LLMService,
        executor: StepExecutor | None = None,
        memory_manager: PlanningMemoryManager | None = None,
        store: RunRecordStore | None = None,
        review_gateway: HumanPlanReviewGateway | None = None,
        scheduler: DAGScheduler | None = None,
        replanner: ReplanningEngine | None = None,
        plan_generator: PlanGenerator | None = None,
        verifier: StepVerifier | None = None,
        reflector: PlanReflector | None = None,
        config: AssistantConfig | None = None,
    ) -> None:
        """Initialize assistant facade with dependencies or sensible defaults."""
        self.llm_service = llm_service
        self.config = config or get_assistant_config()
        self.executor = executor or Module17StepExecutor(config=self.config)
        self.memory_manager = memory_manager or PlanningMemoryManager(
            memory_service=MemoryService(),
            config=self.config,
        )
        self.store = store or FileRunRecordStore(config=self.config)
        self.review_gateway = review_gateway or HumanPlanReviewGateway(config=self.config)
        self.verifier = verifier or StepVerifier(llm_service=llm_service, config=self.config)
        self.scheduler = scheduler or DAGScheduler(
            executor=self.executor,
            verifier=self.verifier,
            config=self.config,
        )
        self.replanner = replanner or ReplanningEngine(
            llm_service=llm_service,
            config=self.config,
        )
        self.plan_generator = plan_generator or PlanGenerator(
            llm_service=llm_service,
            memory_manager=self.memory_manager,
            step_executor=self.executor,
            config=self.config,
        )
        self.reflector = reflector

    async def run(
        self,
        goal: str,
        user_id: str,
        budget: RunBudget | None = None,
        decision_provider: DecisionProvider | None = None,
        context: dict[str, Any] | None = None,
    ) -> AssistantRunResult:
        """Execute the full plan-and-execute lifecycle for a user goal.

        Args:
            goal: The research or operational goal.
            user_id: User identifier for authorization and personalization.
            budget: Optional RunBudget for token/cost/timeout enforcement.
            decision_provider: Optional async callback for human review decisions.
            context: Optional initial contextual dictionary.

        Returns:
            AssistantRunResult containing status, synthesized deliverable, and telemetry.
        """
        run_id = str(uuid.uuid4())
        active_budget = budget or RunBudget(
            max_total_tokens=self.config.max_total_tokens,
            max_total_cost_usd=self.config.max_total_cost_usd,
            total_timeout_seconds=self.config.default_timeout_seconds,
            max_concurrency=self.config.max_concurrency,
            max_planning_retries=self.config.max_planning_retries,
            max_step_retries=self.config.max_step_retries,
            max_replans=self.config.max_replans,
        )
        active_budget.start_timer()

        run_record = RunRecord(
            run_id=run_id,
            goal=goal,
            user_id=user_id,
            status=RunStatus.INITIALIZING,
        )
        await self.store.save_run(run_record)

        safe_log(
            _logger,
            logging.INFO,
            f"PlanAndExecuteAssistant starting run '{run_id}' for goal: '{goal}'",
            event=ObservabilityEvent.TOOL_LOOP_START,
            run_id=run_id,
            user_id=user_id,
        )

        current_plan = Plan(goal=goal, steps=[])

        try:
            # -------------------------------------------------------------
            # Phase 1: Context & Memory Retrieval
            # -------------------------------------------------------------
            active_budget.check_budget()
            user_preferences = await self.memory_manager.get_user_preferences(
                user_id=user_id, query=goal
            )
            capabilities = self.executor.get_available_capabilities()

            # -------------------------------------------------------------
            # Phase 2: Structured Plan Generation & Automated Repair Loop
            # -------------------------------------------------------------
            run_record.status = RunStatus.PLANNING
            await self.store.save_run(run_record)
            active_budget.check_budget()

            initial_plan = await self.plan_generator.generate_plan(
                goal=goal,
                user_id=user_id,
                user_preferences=user_preferences,
                context=context,
                budget=active_budget,
            )
            run_record.plan_versions.append(initial_plan)
            current_plan = initial_plan
            await self.store.save_run(run_record)

            # -------------------------------------------------------------
            # Phase 3: Human Plan Review (Gating)
            # -------------------------------------------------------------
            run_record.status = RunStatus.REVIEW_PENDING
            await self.store.save_run(run_record)
            active_budget.check_budget()

            try:
                approved_plan = await self.review_gateway.review_plan(
                    plan=initial_plan,
                    goal=goal,
                    user_id=user_id,
                    decision_provider=decision_provider,
                )
            except PlanRejectedError as reject_exc:
                run_record.status = RunStatus.CANCELLED
                run_record.error_message = f"Plan rejected by reviewer: {reject_exc}"
                return await self._finalize_run(run_record, current_plan, active_budget)

            current_plan = approved_plan
            if current_plan.version > initial_plan.version:
                run_record.plan_versions.append(current_plan)
            await self.store.save_run(run_record)

            # -------------------------------------------------------------
            # Phase 4: Concurrent Execution & Dynamic Replanning Loop
            # -------------------------------------------------------------
            run_record.status = RunStatus.EXECUTING
            await self.store.save_run(run_record)

            replan_count = 0
            all_step_results: dict[str, str] = {}

            while True:
                active_budget.check_budget()

                sched_result = await self.scheduler.execute_plan(
                    plan=current_plan,
                    user_id=user_id,
                    budget=active_budget,
                    step_records_collector=run_record.step_executions,
                )

                # Merge step execution records (avoiding duplicates if streamed in real time) and cache outputs
                for s_id, recs in sched_result.step_records.items():
                    if s_id not in run_record.step_executions:
                        run_record.step_executions[s_id] = []
                    for r in recs:
                        if r not in run_record.step_executions[s_id]:
                            run_record.step_executions[s_id].append(r)

                all_step_results.update(sched_result.step_results)

                if sched_result.status == StepStatus.COMPLETED:
                    safe_log(
                        _logger,
                        logging.INFO,
                        f"All steps completed successfully for run '{run_id}' (plan version {current_plan.version})",
                        run_id=run_id,
                    )
                    break

                # Step failure encountered after local retries exhausted
                safe_log(
                    _logger,
                    logging.WARNING,
                    f"Step '{sched_result.failed_step_id}' failed in run '{run_id}'. Escalating to replanner...",
                    run_id=run_id,
                    failed_step_id=sched_result.failed_step_id,
                )

                if replan_count >= self.config.max_replans:
                    run_record.status = RunStatus.FAILED
                    run_record.error_message = (
                        f"Replanning ceiling exceeded ({replan_count}/{self.config.max_replans}). "
                        f"Last failure: {sched_result.error_message}"
                    )
                    return await self._finalize_run(run_record, current_plan, active_budget)

                run_record.status = RunStatus.REPLANNING
                await self.store.save_run(run_record)
                active_budget.check_budget()

                replan_res = await self.replanner.replan(
                    current_plan=current_plan,
                    failed_step_id=sched_result.failed_step_id or "unknown",
                    failure_reason=sched_result.error_message or "Step execution failed",
                    step_results=all_step_results,
                    replan_count=replan_count,
                    capabilities=capabilities,
                    user_preferences=user_preferences,
                    budget=active_budget,
                )

                replan_count = replan_res.replan_count
                current_plan = replan_res.new_plan
                run_record.plan_versions.append(current_plan)
                run_record.status = RunStatus.EXECUTING
                await self.store.save_run(run_record)

            # -------------------------------------------------------------
            # Optional Reflection Phase (Module 19 integration)
            # -------------------------------------------------------------
            if self.reflector is not None:
                active_budget.check_budget()
                ref_res = await self.reflector.reflect(
                    plan=current_plan,
                    step_executions=run_record.step_executions,
                    user_id=user_id,
                )
                safe_log(
                    _logger,
                    logging.INFO,
                    f"Reflection analysis completed: should_replan={ref_res.should_replan}, score={ref_res.score}",
                    run_id=run_id,
                    critique=ref_res.critique,
                )

            # -------------------------------------------------------------
            # Phase 5: Synthesis of Final Verified Deliverable
            # -------------------------------------------------------------
            active_budget.check_budget()
            synth_prompt = build_synthesis_prompt(
                goal=goal,
                step_results=all_step_results,
                plan=current_plan,
                user_preferences=user_preferences,
            )

            try:
                from genai_core import LLMManagerRequest

                synth_req = LLMManagerRequest(prompt=synth_prompt)
                raw_synth = await self.llm_service.complete(synth_req)
                synth_text = (
                    getattr(raw_synth, "text", None)
                    or getattr(raw_synth, "content", None)
                    or str(raw_synth)
                )
                # Track token usage from synthesis
                usage = getattr(raw_synth, "usage", {}) or {}
                if isinstance(usage, dict):
                    tokens = int(usage.get("output_tokens", 0)) + int(usage.get("input_tokens", 0))
                else:
                    tokens = getattr(usage, "total_tokens", 0)
                cost = getattr(raw_synth, "cost_usd", 0.0)
                active_budget.record_usage(tokens=tokens, cost_usd=cost)
            except Exception as synth_exc:  # noqa: BLE001
                safe_log(
                    _logger,
                    logging.WARNING,
                    f"Deliverable synthesis LLM call failed: {synth_exc}. Falling back to concatenated summaries.",
                    error=str(synth_exc),
                )
                synth_text = "\n\n".join(
                    f"### Step {s_id}\n{summary}" for s_id, summary in all_step_results.items()
                )

            run_record.final_deliverable = synth_text
            run_record.status = RunStatus.COMPLETED

            # Record completed episode in episodic memory (non-fatal telemetry)
            try:
                await self.memory_manager.record_plan_episode(
                    user_id=user_id,
                    goal=goal,
                    plan=current_plan,
                    result_summary=synth_text,
                    metadata={
                        "total_tokens": active_budget.current_tokens,
                        "total_cost_usd": active_budget.current_cost_usd,
                    },
                )
            except Exception as mem_exc:  # noqa: BLE001
                safe_log(
                    _logger,
                    logging.WARNING,
                    f"Failed to record episodic memory for run '{run_id}': {mem_exc}",
                    run_id=run_id,
                    error=str(mem_exc),
                )

            return await self._finalize_run(run_record, current_plan, active_budget)

        except BudgetExceededError as budget_exc:
            safe_log(
                _logger,
                logging.ERROR,
                f"Run '{run_id}' halted: budget exceeded ({budget_exc})",
                run_id=run_id,
                error=str(budget_exc),
            )
            run_record.status = RunStatus.FAILED
            run_record.error_message = f"Budget exceeded: {budget_exc}"
            return await self._finalize_run(run_record, current_plan, active_budget)
        except Exception as exc:  # noqa: BLE001
            safe_log(
                _logger,
                logging.ERROR,
                f"Run '{run_id}' encountered unexpected error: {exc}",
                run_id=run_id,
                error=str(exc),
            )
            run_record.status = RunStatus.FAILED
            run_record.error_message = f"Execution error: {exc}"
            return await self._finalize_run(run_record, current_plan, active_budget)

    async def _finalize_run(
        self,
        run_record: RunRecord,
        plan: Plan,
        budget: RunBudget,
    ) -> AssistantRunResult:
        """Stamp duration, token, and cost telemetry, persist RunRecord, and construct result."""
        now_monotonic = time.monotonic()
        start = budget.start_monotonic or now_monotonic
        duration_ms = round((now_monotonic - start) * 1000.0, 2)

        run_record.total_tokens = budget.current_tokens
        run_record.total_cost_usd = budget.current_cost_usd
        run_record.duration_ms = duration_ms
        run_record.completed_at = datetime.now(UTC)

        # Asynchronously persist final snapshot via RunRecordStore protocol
        try:
            await self.store.save_run(run_record)
        except Exception as store_exc:  # noqa: BLE001
            safe_log(
                _logger,
                logging.WARNING,
                f"Failed to asynchronously save run record '{run_record.run_id}': {store_exc}",
                error=str(store_exc),
            )

        return AssistantRunResult(
            run_id=run_record.run_id,
            status=run_record.status,
            final_deliverable=run_record.final_deliverable,
            plan=plan,
            run_record=run_record,
            total_tokens=run_record.total_tokens,
            total_cost_usd=run_record.total_cost_usd,
            duration_ms=run_record.duration_ms,
            error_message=run_record.error_message,
        )
