"""Parallel research worker scheduler with concurrency bounding and failure isolation."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from genai_core.observability import get_logger

from ..core.budget import SharedRunBudget
from ..core.config import ResearchTeamConfig, get_research_team_config
from ..models import (
    CheckpointState,
    WorkerBrief,
    WorkerExecutionResult,
    WorkerStatus,
)
from ..storage.protocol import CheckpointStore
from ..workers.registry import WorkerRegistry, get_default_worker_registry
from .validator import WorkerOutputValidator

_logger = get_logger("parallel_research_team.scheduling.scheduler")


class ParallelResearchScheduler:
    """Coordinates concurrent research worker dispatch, failure isolation, and real-time checkpointing."""

    def __init__(
        self,
        config: ResearchTeamConfig | None = None,
        worker_registry: WorkerRegistry | None = None,
        validator: WorkerOutputValidator | None = None,
        checkpoint_store: CheckpointStore | None = None,
        retrieval_service: Any | None = None,
        llm_service: Any | None = None,
    ) -> None:
        self.config = config or get_research_team_config()
        self.worker_registry = worker_registry or get_default_worker_registry()
        self.validator = validator or WorkerOutputValidator()
        self.checkpoint_store = checkpoint_store
        self.retrieval_service = retrieval_service
        self.llm_service = llm_service

    def _detect_cycles(self, briefs: list[WorkerBrief]) -> bool:
        """Detect whether dependency graph between briefs contains a cycle."""
        adj: dict[str, list[str]] = {b.brief_id: list(b.depends_on) for b in briefs}
        visited: dict[str, int] = {}  # 0: unvisited, 1: visiting, 2: visited

        def has_cycle(node: str) -> bool:
            state = visited.get(node, 0)
            if state == 1:
                return True
            if state == 2:
                return False
            visited[node] = 1
            for neighbor in adj.get(node, []):
                if has_cycle(neighbor):
                    return True
            visited[node] = 2
            return False

        return any(has_cycle(b.brief_id) for b in briefs if visited.get(b.brief_id, 0) == 0)

    async def schedule_and_execute(
        self,
        briefs: list[WorkerBrief],
        shared_budget: SharedRunBudget,
        checkpoint: CheckpointState | None = None,
    ) -> dict[str, WorkerExecutionResult]:
        """Dispatch worker briefs in parallel subject to concurrency limits and failure isolation."""
        if not briefs:
            _logger.info("No worker briefs provided to schedule")
            return {}

        # 1. Validate dependency graph for circular dependencies
        if self._detect_cycles(briefs):
            _logger.error("Circular dependency detected in worker briefs")
            now = datetime.now(UTC)
            return {
                b.brief_id: WorkerExecutionResult(
                    brief_id=b.brief_id,
                    status=WorkerStatus.FAILED,
                    error_message="Circular dependency detected among worker briefs.",
                    started_at=now,
                    completed_at=now,
                )
                for b in briefs
            }

        # 2. Concurrency envelope and state locks
        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        state_lock = asyncio.Lock()
        results: dict[str, WorkerExecutionResult] = {}
        completed_events: dict[str, asyncio.Event] = {b.brief_id: asyncio.Event() for b in briefs}

        _logger.info(
            "Scheduling %d worker briefs with max_concurrency=%d",
            len(briefs),
            self.config.max_concurrency,
        )

        async def _execute_single_brief(brief: WorkerBrief) -> WorkerExecutionResult:
            try:
                # A. Wait for upstream dependency completions if specified
                if brief.depends_on:
                    for dep_id in brief.depends_on:
                        if dep_id in completed_events:
                            await completed_events[dep_id].wait()

                # B. Acquire concurrency slot
                async with semaphore:
                    # Proactive budget check before worker instantiation
                    if not await shared_budget.can_proceed():
                        res = WorkerExecutionResult(
                            brief_id=brief.brief_id,
                            status=WorkerStatus.FAILED,
                            error_message="Shared run budget already exhausted before worker execution.",
                            started_at=datetime.now(UTC),
                            completed_at=datetime.now(UTC),
                        )
                    else:
                        try:
                            worker_kwargs: dict[str, Any] = {}
                            if self.retrieval_service is not None:
                                worker_kwargs["retrieval_service"] = self.retrieval_service
                            if self.llm_service is not None:
                                worker_kwargs["llm_service"] = self.llm_service

                            worker = self.worker_registry.create_worker(
                                brief=brief,
                                config=self.config,
                                **worker_kwargs,
                            )
                            raw_result = await worker.execute(
                                brief=brief,
                                shared_budget=shared_budget,
                            )
                            res = self.validator.validate(raw_result, brief)
                        except Exception as exc:
                            _logger.exception(
                                "Unexpected failure in worker execution for brief '%s'",
                                brief.brief_id,
                            )
                            res = WorkerExecutionResult(
                                brief_id=brief.brief_id,
                                status=WorkerStatus.FAILED,
                                error_message=f"Unhandled worker failure: {exc}",
                                started_at=datetime.now(UTC),
                                completed_at=datetime.now(UTC),
                            )

                    # C. Real-time incremental state update & atomic checkpoint persistence
                    async with state_lock:
                        results[brief.brief_id] = res

                        if checkpoint is not None:
                            checkpoint.worker_results[brief.brief_id] = res

                            # Accumulate authoritative sources in real time
                            if res.output and res.output.sources:
                                for src in res.output.sources:
                                    s_id = str(src.get("source_id", "")).strip()
                                    c_id = str(src.get("chunk_id", "")).strip()
                                    if s_id and c_id:
                                        checkpoint.authoritative_sources[f"{s_id}#{c_id}"] = src
                                    elif c_id:
                                        checkpoint.authoritative_sources[c_id] = src

                            checkpoint.used_tokens = shared_budget.current_tokens
                            checkpoint.used_cost_usd = shared_budget.current_cost_usd
                            checkpoint.elapsed_seconds = shared_budget.elapsed_seconds
                            checkpoint.updated_at = datetime.now(UTC)

                            if self.checkpoint_store is not None:
                                try:
                                    await self.checkpoint_store.save_checkpoint(checkpoint)
                                except Exception as cp_err:  # noqa: BLE001
                                    _logger.warning(
                                        "Failed to persist real-time checkpoint for brief '%s': %s",
                                        brief.brief_id,
                                        cp_err,
                                    )

                    return res
            finally:
                # Signal dependent workers even on cancellation, failure, or exception
                completed_events[brief.brief_id].set()

        # 3. Dispatch all tasks concurrently with exception shielding
        tasks = [_execute_single_brief(b) for b in briefs]
        raw_outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        # 4. Final verification: ensure every brief has a recorded result
        for idx, outcome in enumerate(raw_outcomes):
            brief_id = briefs[idx].brief_id
            if isinstance(outcome, BaseException):
                _logger.error(
                    "Task for brief '%s' raised unhandled exception during gather: %s",
                    brief_id,
                    outcome,
                )
                if brief_id not in results:
                    results[brief_id] = WorkerExecutionResult(
                        brief_id=brief_id,
                        status=WorkerStatus.FAILED,
                        error_message=f"Task aborted with exception: {outcome}",
                        started_at=datetime.now(UTC),
                        completed_at=datetime.now(UTC),
                    )

        return results
