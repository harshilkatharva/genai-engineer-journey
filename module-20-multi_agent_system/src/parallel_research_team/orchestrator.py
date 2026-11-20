"""Parallel Research Team orchestrator facade coordinating decomposition, scheduling, synthesis, and verification."""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from typing import Any

from genai_core.observability import get_logger

from .core.budget import SharedRunBudget
from .core.config import ResearchTeamConfig, get_research_team_config
from .decomposition.engine import DecompositionEngine
from .exceptions import BudgetExceededError
from .models import (
    CheckpointState,
    EffortLevel,
    ExecutionMetrics,
    Report,
    ResearchTeamRunResult,
    RunStatus,
    WorkerExecutionResult,
    WorkerStatus,
)
from .scheduling.scheduler import ParallelResearchScheduler
from .storage.file_store import FileCheckpointStore
from .storage.protocol import CheckpointStore
from .synthesis.engine import SynthesisEngine
from .synthesis.source_pool import AuthoritativeSourcePool
from .synthesis.verifier import ResearchVerifier
from .telemetry.pricing import PricingCalculator
from .telemetry.tracker import ResearchTelemetryTracker

_logger = get_logger("parallel_research_team.orchestrator")


def _to_int(val: Any) -> int:
    return val if isinstance(val, int) and not isinstance(val, bool) else 0


def _to_float(val: Any) -> float:
    return float(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else 0.0


class ParallelResearchTeam:
    """Unified facade orchestrating parallel multi-agent research runs."""

    def __init__(
        self,
        config: ResearchTeamConfig | None = None,
        llm_service: Any | None = None,
        retrieval_service: Any | None = None,
        decomposition_engine: DecompositionEngine | None = None,
        scheduler: ParallelResearchScheduler | None = None,
        synthesis_engine: SynthesisEngine | None = None,
        verifier: ResearchVerifier | None = None,
        checkpoint_store: CheckpointStore | None = None,
        memory_service: Any | None = None,
        pricing_calculator: PricingCalculator | None = None,
    ) -> None:
        self.config = config or get_research_team_config()
        self.pricing_calculator = pricing_calculator or PricingCalculator()
        self.checkpoint_store = checkpoint_store or FileCheckpointStore(config=self.config)
        self.memory_service = memory_service
        self.decomposition_engine = decomposition_engine or DecompositionEngine(
            config=self.config,
            llm_service=llm_service,
            memory_service=memory_service,
        )
        self.scheduler = scheduler or ParallelResearchScheduler(
            config=self.config,
            checkpoint_store=self.checkpoint_store,
            retrieval_service=retrieval_service,
            llm_service=llm_service,
        )
        self.synthesis_engine = synthesis_engine or SynthesisEngine(
            config=self.config,
            llm_service=llm_service,
            pricing_calculator=self.pricing_calculator,
        )
        self.verifier = verifier or ResearchVerifier(
            config=self.config,
            judge_llm_service=llm_service,
        )

    async def run(
        self,
        topic: str,
        user_id: str = "default_user",
        session_id: str | None = None,
        effort_override: EffortLevel | None = None,
    ) -> ResearchTeamRunResult:
        """Execute a complete end-to-end research run."""
        clean_topic = topic.strip() if topic else ""
        if not clean_topic:
            raise ValueError("Research topic cannot be empty or pure whitespace.")

        run_id = f"run_{uuid.uuid4().hex[:12]}"
        sess_id = session_id or f"sess_{uuid.uuid4().hex[:8]}"

        budget = SharedRunBudget(config=self.config)
        budget.start_timer()

        tracker = ResearchTelemetryTracker(pricing_calculator=self.pricing_calculator)
        source_pool = AuthoritativeSourcePool()

        checkpoint = CheckpointState(
            run_id=run_id,
            user_id=user_id,
            topic=clean_topic,
            status=RunStatus.INITIALIZING,
        )
        await self.checkpoint_store.save_checkpoint(checkpoint)

        # 1. Decomposition Stage
        checkpoint.status = RunStatus.DECOMPOSING
        await self.checkpoint_store.save_checkpoint(checkpoint)

        t_decomp = time.perf_counter()
        try:
            decomposition = await self.decomposition_engine.decompose(
                topic=clean_topic,
                user_id=user_id,
                session_id=sess_id,
                effort_override=effort_override,
                shared_budget=budget,
            )
        except Exception as exc:
            _logger.exception("Decomposition failed for run %s", run_id)
            checkpoint.status = RunStatus.FAILED
            checkpoint.error_message = f"Decomposition failed: {exc}"
            await self.checkpoint_store.save_checkpoint(checkpoint)
            metrics = tracker.finish()
            return ResearchTeamRunResult(
                run_id=run_id,
                status=RunStatus.FAILED,
                metrics=metrics,
                checkpoint_state=checkpoint,
                error_message=checkpoint.error_message,
            )

        decomp_latency = (time.perf_counter() - t_decomp) * 1000.0
        decomp_in = _to_int(getattr(self.decomposition_engine, "last_input_tokens", 0))
        decomp_out = _to_int(getattr(self.decomposition_engine, "last_output_tokens", 0))
        decomp_tokens = _to_int(
            getattr(self.decomposition_engine, "last_total_tokens", decomp_in + decomp_out)
        )
        decomp_cost = _to_float(getattr(self.decomposition_engine, "last_cost_usd", 0.0))

        tracker.record_decomposition(
            tokens=decomp_tokens,
            cost_usd=decomp_cost,
            latency_ms=decomp_latency,
            model=self.config.default_model,
        )

        checkpoint.effort_level = decomposition.effort_level
        checkpoint.decomposition = decomposition
        checkpoint.status = RunStatus.EXECUTING
        checkpoint.used_tokens = budget.current_tokens
        checkpoint.used_cost_usd = budget.current_cost_usd
        checkpoint.elapsed_seconds = budget.elapsed_seconds
        await self.checkpoint_store.save_checkpoint(checkpoint)

        # 2. Worker Scheduling Stage
        budget_exceeded = False
        budget_warning: str | None = None
        worker_results: dict[str, WorkerExecutionResult] = {}

        try:
            worker_results = await self.scheduler.schedule_and_execute(
                briefs=decomposition.worker_briefs,
                shared_budget=budget,
                checkpoint=checkpoint,
            )
        except BudgetExceededError as exc:
            budget_exceeded = True
            budget_warning = (
                f"Research execution was halted before all briefs could complete because the configured "
                f"resource budget was exhausted ({exc}). Findings gathered prior to budget exhaustion have been preserved."
            )
            _logger.warning("Run %s halted due to budget limit: %s", run_id, exc)

        for res in worker_results.values():
            tracker.record_worker_result(res, model=self.config.default_model)
            if res.output and res.output.sources:
                source_pool.add_sources(res.output.sources)

        checkpoint.authoritative_sources = source_pool.to_dict()
        checkpoint.worker_results = worker_results
        checkpoint.used_tokens = budget.current_tokens
        checkpoint.used_cost_usd = budget.current_cost_usd
        checkpoint.elapsed_seconds = budget.elapsed_seconds

        # 3. Synthesis Stage
        has_usable_evidence = len(source_pool) > 0 or any(
            r.status in (WorkerStatus.COMPLETED, WorkerStatus.PARTIAL)
            for r in worker_results.values()
        )

        if not has_usable_evidence:
            final_status = RunStatus.FAILED
            err = (
                budget_warning
                or "All research workers failed and zero authoritative sources were collected."
            )
            checkpoint.status = final_status
            checkpoint.error_message = err
            await self.checkpoint_store.save_checkpoint(checkpoint)
            metrics = tracker.finish()
            return ResearchTeamRunResult(
                run_id=run_id,
                status=final_status,
                decomposition=decomposition,
                metrics=metrics,
                checkpoint_state=checkpoint,
                error_message=err,
            )

        checkpoint.status = RunStatus.SYNTHESIZING
        await self.checkpoint_store.save_checkpoint(checkpoint)

        report: Report | None = None
        t_synth = time.perf_counter()
        try:
            report = await self.synthesis_engine.synthesize(
                topic=clean_topic,
                worker_results=worker_results,
                source_pool=source_pool,
                shared_budget=budget,
            )
            synth_latency = (time.perf_counter() - t_synth) * 1000.0
            synth_in = _to_int(getattr(self.synthesis_engine, "last_input_tokens", 0))
            synth_out = _to_int(getattr(self.synthesis_engine, "last_output_tokens", 0))
            synth_tokens = _to_int(
                getattr(self.synthesis_engine, "last_total_tokens", synth_in + synth_out)
            )
            synth_cost = _to_float(getattr(self.synthesis_engine, "last_cost_usd", 0.0))

            tracker.record_synthesis(
                tokens=synth_tokens,
                cost_usd=synth_cost,
                latency_ms=synth_latency,
                model=self.config.synthesis_model,
            )
        except BudgetExceededError as exc:
            budget_exceeded = True
            budget_warning = (
                f"Report synthesis halted due to budget limit ({exc}). Partial findings preserved."
            )
            _logger.warning("Synthesis budget exceeded: %s", exc)
        except Exception as exc:
            _logger.exception("Synthesis error for run %s", run_id)
            checkpoint.status = RunStatus.FAILED
            checkpoint.error_message = f"Synthesis failed: {exc}"
            await self.checkpoint_store.save_checkpoint(checkpoint)
            metrics = tracker.finish()
            return ResearchTeamRunResult(
                run_id=run_id,
                status=RunStatus.FAILED,
                decomposition=decomposition,
                metrics=metrics,
                checkpoint_state=checkpoint,
                error_message=checkpoint.error_message,
            )

        if budget_warning and report:
            if budget_warning not in report.limitations:
                report.limitations.append(budget_warning)
            if budget_warning not in report.unresolved_limitations:
                report.unresolved_limitations.append(budget_warning)

        # 4. Verification Stage
        if report:
            checkpoint.status = RunStatus.VERIFYING
            await self.checkpoint_store.save_checkpoint(checkpoint)
            verification_result = await self.verifier.verify(report=report, source_pool=source_pool)
            tracker.record_verification(
                passed=verification_result.passed,
                gate1_passed=verification_result.gate1_passed,
                gate2_passed=verification_result.gate2_passed,
                issues_count=len(verification_result.issues),
                summary=verification_result.summary,
            )
            if not verification_result.passed:
                _logger.warning(
                    "Report verification identified issues: %s", verification_result.summary
                )
                verif_note = f"Verification advisory: {verification_result.summary}"
                if verif_note not in report.limitations:
                    report.limitations.append(verif_note)

        # 5. Final Outcome Determination
        if budget_exceeded:
            final_status = RunStatus.PARTIALLY_COMPLETED
        elif report is not None:
            any_failures = any(
                r.status in (WorkerStatus.FAILED, WorkerStatus.TIMEOUT, WorkerStatus.PARTIAL)
                for r in worker_results.values()
            )
            final_status = RunStatus.PARTIALLY_COMPLETED if any_failures else RunStatus.COMPLETED
        else:
            final_status = RunStatus.FAILED

        checkpoint.status = final_status
        checkpoint.final_report = report.model_dump() if report else None
        checkpoint.used_tokens = budget.current_tokens
        checkpoint.used_cost_usd = budget.current_cost_usd
        checkpoint.elapsed_seconds = budget.elapsed_seconds
        checkpoint.updated_at = datetime.now(UTC)
        await self.checkpoint_store.save_checkpoint(checkpoint)

        metrics = tracker.finish()

        # Optional Memory Persistence
        if self.memory_service and report:
            try:
                summary_text = f"Research Topic: {clean_topic} | Title: {report.title} | Status: {final_status.value}"
                await self.memory_service.record(
                    user_id=user_id,
                    content=summary_text,
                    metadata={"run_id": run_id, "status": final_status.value},
                )
            except Exception as mem_err:  # noqa: BLE001
                _logger.warning("Failed to record research outcome to memory: %s", mem_err)

        return ResearchTeamRunResult(
            run_id=run_id,
            status=final_status,
            report=report,
            decomposition=decomposition,
            metrics=metrics,
            checkpoint_state=checkpoint,
            error_message=budget_warning,
        )

    async def resume(self, run_id: str) -> ResearchTeamRunResult:
        """Resume execution of an interrupted or partially completed research run from checkpoint."""
        checkpoint = await self.checkpoint_store.get_checkpoint(run_id)
        if not checkpoint:
            return ResearchTeamRunResult(
                run_id=run_id,
                status=RunStatus.FAILED,
                error_message=f"Checkpoint '{run_id}' not found.",
            )

        if checkpoint.status == RunStatus.COMPLETED and checkpoint.final_report:
            cached_report = Report.model_validate(checkpoint.final_report)
            metrics = ExecutionMetrics(
                total_tokens=checkpoint.used_tokens,
                total_cost_usd=checkpoint.used_cost_usd,
                duration_ms=checkpoint.elapsed_seconds * 1000.0,
            )
            return ResearchTeamRunResult(
                run_id=run_id,
                status=RunStatus.COMPLETED,
                report=cached_report,
                decomposition=checkpoint.decomposition,
                metrics=metrics,
                checkpoint_state=checkpoint,
            )

        if not checkpoint.decomposition:
            _logger.info("Resuming run %s before decomposition completed: re-running", run_id)
            return await self.run(topic=checkpoint.topic, user_id=checkpoint.user_id)

        # Restore evidence and state
        source_pool = AuthoritativeSourcePool.from_dict(checkpoint.authoritative_sources)
        for r in checkpoint.worker_results.values():
            if r.output and r.output.sources:
                source_pool.add_sources(r.output.sources)

        all_briefs = checkpoint.decomposition.worker_briefs
        completed_brief_ids = {
            bid
            for bid, r in checkpoint.worker_results.items()
            if r.status in (WorkerStatus.COMPLETED, WorkerStatus.PARTIAL)
        }
        pending_briefs = [b for b in all_briefs if b.brief_id not in completed_brief_ids]

        budget = SharedRunBudget(
            config=self.config,
            initial_tokens=checkpoint.used_tokens,
            initial_cost_usd=checkpoint.used_cost_usd,
        )
        budget.start_timer()

        tracker = ResearchTelemetryTracker(
            pricing_calculator=self.pricing_calculator,
            initial_tokens=checkpoint.used_tokens,
            initial_cost_usd=checkpoint.used_cost_usd,
        )
        for r in checkpoint.worker_results.values():
            tracker.record_worker_result(r, model=self.config.default_model)

        budget_exceeded = False
        budget_warning: str | None = None

        if pending_briefs:
            checkpoint.status = RunStatus.EXECUTING
            await self.checkpoint_store.save_checkpoint(checkpoint)
            try:
                new_results = await self.scheduler.schedule_and_execute(
                    briefs=pending_briefs,
                    shared_budget=budget,
                    checkpoint=checkpoint,
                )
                checkpoint.worker_results.update(new_results)
                for res in new_results.values():
                    tracker.record_worker_result(res, model=self.config.default_model)
                    if res.output and res.output.sources:
                        source_pool.add_sources(res.output.sources)
            except BudgetExceededError as exc:
                budget_exceeded = True
                budget_warning = f"Resumed execution halted due to budget limit ({exc}). Completed work preserved."
                _logger.warning("Resumed run %s budget exceeded: %s", run_id, exc)

        checkpoint.authoritative_sources = source_pool.to_dict()
        checkpoint.used_tokens = budget.current_tokens
        checkpoint.used_cost_usd = budget.current_cost_usd
        checkpoint.elapsed_seconds = budget.elapsed_seconds

        # Proceed to Synthesis
        checkpoint.status = RunStatus.SYNTHESIZING
        await self.checkpoint_store.save_checkpoint(checkpoint)

        report: Report | None = None
        t_synth = time.perf_counter()
        try:
            report = await self.synthesis_engine.synthesize(
                topic=checkpoint.topic,
                worker_results=checkpoint.worker_results,
                source_pool=source_pool,
                shared_budget=budget,
            )
            synth_latency = (time.perf_counter() - t_synth) * 1000.0
            synth_in = _to_int(getattr(self.synthesis_engine, "last_input_tokens", 0))
            synth_out = _to_int(getattr(self.synthesis_engine, "last_output_tokens", 0))
            synth_tokens = _to_int(
                getattr(self.synthesis_engine, "last_total_tokens", synth_in + synth_out)
            )
            synth_cost = _to_float(getattr(self.synthesis_engine, "last_cost_usd", 0.0))

            tracker.record_synthesis(
                tokens=synth_tokens,
                cost_usd=synth_cost,
                latency_ms=synth_latency,
                model=self.config.synthesis_model,
            )
        except BudgetExceededError as exc:
            budget_exceeded = True
            budget_warning = f"Resumed synthesis halted due to budget limit ({exc})."
            _logger.warning("Resumed synthesis budget exceeded: %s", exc)
        except Exception as exc:
            _logger.exception("Resumed synthesis failed")
            checkpoint.status = RunStatus.FAILED
            checkpoint.error_message = f"Synthesis failed: {exc}"
            await self.checkpoint_store.save_checkpoint(checkpoint)
            metrics = tracker.finish()
            return ResearchTeamRunResult(
                run_id=run_id,
                status=RunStatus.FAILED,
                decomposition=checkpoint.decomposition,
                metrics=metrics,
                checkpoint_state=checkpoint,
                error_message=checkpoint.error_message,
            )

        if budget_warning and report:
            if budget_warning not in report.limitations:
                report.limitations.append(budget_warning)
            if budget_warning not in report.unresolved_limitations:
                report.unresolved_limitations.append(budget_warning)

        # Verification
        if report:
            checkpoint.status = RunStatus.VERIFYING
            await self.checkpoint_store.save_checkpoint(checkpoint)
            verification_result = await self.verifier.verify(report=report, source_pool=source_pool)
            tracker.record_verification(
                passed=verification_result.passed,
                gate1_passed=verification_result.gate1_passed,
                gate2_passed=verification_result.gate2_passed,
                issues_count=len(verification_result.issues),
                summary=verification_result.summary,
            )
            if not verification_result.passed:
                verif_note = f"Verification advisory: {verification_result.summary}"
                if verif_note not in report.limitations:
                    report.limitations.append(verif_note)

        # Final Status
        if budget_exceeded:
            final_status = RunStatus.PARTIALLY_COMPLETED
        elif report is not None:
            any_failures = any(
                r.status in (WorkerStatus.FAILED, WorkerStatus.TIMEOUT, WorkerStatus.PARTIAL)
                for r in checkpoint.worker_results.values()
            )
            final_status = RunStatus.PARTIALLY_COMPLETED if any_failures else RunStatus.COMPLETED
        else:
            final_status = RunStatus.FAILED

        checkpoint.status = final_status
        checkpoint.final_report = report.model_dump() if report else None
        checkpoint.used_tokens = budget.current_tokens
        checkpoint.used_cost_usd = budget.current_cost_usd
        checkpoint.elapsed_seconds = budget.elapsed_seconds
        checkpoint.updated_at = datetime.now(UTC)
        await self.checkpoint_store.save_checkpoint(checkpoint)

        metrics = tracker.finish()

        return ResearchTeamRunResult(
            run_id=run_id,
            status=final_status,
            report=report,
            decomposition=checkpoint.decomposition,
            metrics=metrics,
            checkpoint_state=checkpoint,
            error_message=budget_warning,
        )
