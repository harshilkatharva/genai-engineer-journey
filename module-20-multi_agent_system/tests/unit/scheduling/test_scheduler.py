"""Unit tests for ParallelResearchScheduler."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from parallel_research_team.core.budget import SharedRunBudget
from parallel_research_team.core.config import ResearchTeamConfig
from parallel_research_team.models import (
    CheckpointState,
    RunStatus,
    WorkerBrief,
    WorkerExecutionResult,
    WorkerResearchOutput,
    WorkerStatus,
)
from parallel_research_team.scheduling.scheduler import ParallelResearchScheduler
from parallel_research_team.workers.protocol import ResearchWorkerProtocol
from parallel_research_team.workers.registry import WorkerRegistry


class MockControlledWorker(ResearchWorkerProtocol):
    """Test worker with configurable delay, failure flag, and source yield."""

    def __init__(
        self,
        delay: float = 0.05,
        should_fail: bool = False,
        sources: list[dict] | None = None,
    ) -> None:
        self.delay = delay
        self.should_fail = should_fail
        self.sources = sources or []

    @property
    def worker_type(self) -> str:
        return "mock"

    async def execute(
        self,
        brief: WorkerBrief,
        shared_budget: SharedRunBudget,
    ) -> WorkerExecutionResult:
        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if self.should_fail:
            raise RuntimeError(f"Worker {brief.brief_id} deliberate failure")

        await shared_budget.record_usage(tokens=100, cost_usd=0.001)

        output = WorkerResearchOutput(
            brief_id=brief.brief_id,
            findings_summary=f"Findings for {brief.brief_id}",
            sources=self.sources,
            claims=[],
        )
        return WorkerExecutionResult(
            brief_id=brief.brief_id,
            status=WorkerStatus.COMPLETED,
            output=output,
            tokens_used=100,
            cost_usd=0.001,
        )


@pytest.mark.asyncio
async def test_scheduler_parallel_dispatch_and_concurrency(
    test_config: ResearchTeamConfig,
) -> None:
    test_config.max_concurrency = 2
    budget = SharedRunBudget(config=test_config, max_total_tokens=10000, max_total_cost_usd=1.0)

    registry = WorkerRegistry()
    registry.register("mock", lambda brief, config, **kwargs: MockControlledWorker(delay=0.02))

    scheduler = ParallelResearchScheduler(config=test_config, worker_registry=registry)

    briefs = [
        WorkerBrief(
            brief_id=f"b_{i}",
            title=f"Brief {i}",
            topic="Topic",
            objective="Obj",
            task_boundaries=["In", "Out"],
            worker_type="mock",
        )
        for i in range(4)
    ]

    results = await scheduler.schedule_and_execute(briefs=briefs, shared_budget=budget)

    assert len(results) == 4
    for b in briefs:
        assert results[b.brief_id].status == WorkerStatus.COMPLETED
    assert budget.current_tokens == 400


@pytest.mark.asyncio
async def test_scheduler_strict_failure_isolation(test_config: ResearchTeamConfig) -> None:
    budget = SharedRunBudget(config=test_config, max_total_tokens=10000, max_total_cost_usd=1.0)
    registry = WorkerRegistry()

    # Worker b_fail raises an exception, while b_ok1 and b_ok2 succeed
    def factory(brief: WorkerBrief, config: ResearchTeamConfig, **kwargs) -> MockControlledWorker:
        fail = brief.brief_id == "b_fail"
        return MockControlledWorker(delay=0.02, should_fail=fail)

    registry.register("mock", factory)
    scheduler = ParallelResearchScheduler(config=test_config, worker_registry=registry)

    briefs = [
        WorkerBrief(
            brief_id="b_ok1",
            title="Ok 1",
            topic="Topic",
            objective="Obj",
            task_boundaries=["In", "Out"],
            worker_type="mock",
        ),
        WorkerBrief(
            brief_id="b_fail",
            title="Fail",
            topic="Topic",
            objective="Obj",
            task_boundaries=["In", "Out"],
            worker_type="mock",
        ),
        WorkerBrief(
            brief_id="b_ok2",
            title="Ok 2",
            topic="Topic",
            objective="Obj",
            task_boundaries=["In", "Out"],
            worker_type="mock",
        ),
    ]

    results = await scheduler.schedule_and_execute(briefs=briefs, shared_budget=budget)

    # STRICT FAILURE ISOLATION: b_ok1 and b_ok2 succeed despite b_fail crashing
    assert results["b_ok1"].status == WorkerStatus.COMPLETED
    assert results["b_ok2"].status == WorkerStatus.COMPLETED
    assert results["b_fail"].status == WorkerStatus.FAILED
    assert "deliberate failure" in (results["b_fail"].error_message or "")


@pytest.mark.asyncio
async def test_scheduler_real_time_checkpointing(test_config: ResearchTeamConfig) -> None:
    budget = SharedRunBudget(config=test_config, max_total_tokens=10000, max_total_cost_usd=1.0)
    registry = WorkerRegistry()

    sample_source = {
        "source_id": "src_1",
        "chunk_id": "chk_101",
        "title": "Source Doc",
        "content": "Valuable research content.",
    }
    registry.register(
        "mock",
        lambda brief, config, **kwargs: MockControlledWorker(delay=0.01, sources=[sample_source]),
    )

    mock_checkpoint_store = AsyncMock()
    mock_checkpoint_store.save_checkpoint = AsyncMock()

    scheduler = ParallelResearchScheduler(
        config=test_config,
        worker_registry=registry,
        checkpoint_store=mock_checkpoint_store,
    )

    checkpoint = CheckpointState(
        run_id="run_test_cp",
        user_id="user_1",
        topic="Test Topic",
        status=RunStatus.EXECUTING,
    )

    briefs = [
        WorkerBrief(
            brief_id="b1",
            title="Brief 1",
            topic="Topic",
            objective="Obj",
            task_boundaries=["In", "Out"],
            worker_type="mock",
        ),
        WorkerBrief(
            brief_id="b2",
            title="Brief 2",
            topic="Topic",
            objective="Obj",
            task_boundaries=["In", "Out"],
            worker_type="mock",
        ),
    ]

    results = await scheduler.schedule_and_execute(
        briefs=briefs,
        shared_budget=budget,
        checkpoint=checkpoint,
    )

    assert len(results) == 2
    # Checkpoint store must have been invoked after each worker completion
    assert mock_checkpoint_store.save_checkpoint.call_count >= 2
    assert "chk_101" in checkpoint.authoritative_sources
    assert checkpoint.used_tokens == 200


@pytest.mark.asyncio
async def test_scheduler_dependency_chain(test_config: ResearchTeamConfig) -> None:
    budget = SharedRunBudget(config=test_config, max_total_tokens=10000, max_total_cost_usd=1.0)
    registry = WorkerRegistry()
    execution_order: list[str] = []

    class OrderTrackingWorker(ResearchWorkerProtocol):
        @property
        def worker_type(self) -> str:
            return "order"

        async def execute(
            self, brief: WorkerBrief, shared_budget: SharedRunBudget
        ) -> WorkerExecutionResult:
            execution_order.append(f"start_{brief.brief_id}")
            await asyncio.sleep(0.02)
            execution_order.append(f"end_{brief.brief_id}")
            output = WorkerResearchOutput(
                brief_id=brief.brief_id,
                findings_summary=f"Findings for {brief.brief_id}",
                sources=[],
                claims=[],
            )
            return WorkerExecutionResult(
                brief_id=brief.brief_id,
                status=WorkerStatus.COMPLETED,
                output=output,
            )

    registry.register("order", lambda brief, config, **kwargs: OrderTrackingWorker())
    scheduler = ParallelResearchScheduler(config=test_config, worker_registry=registry)

    # b_dep depends on b_first
    briefs = [
        WorkerBrief(
            brief_id="b_first",
            title="First",
            topic="Topic",
            objective="Obj",
            task_boundaries=["In", "Out"],
            worker_type="order",
        ),
        WorkerBrief(
            brief_id="b_dep",
            title="Dependent",
            topic="Topic",
            objective="Obj",
            task_boundaries=["In", "Out"],
            depends_on=["b_first"],
            worker_type="order",
        ),
    ]

    results = await scheduler.schedule_and_execute(briefs=briefs, shared_budget=budget)
    assert results["b_first"].status == WorkerStatus.COMPLETED
    assert results["b_dep"].status == WorkerStatus.COMPLETED

    # Verify execution order: b_first completes before b_dep starts
    first_end_idx = execution_order.index("end_b_first")
    dep_start_idx = execution_order.index("start_b_dep")
    assert first_end_idx < dep_start_idx


@pytest.mark.asyncio
async def test_scheduler_circular_dependency_detected(test_config: ResearchTeamConfig) -> None:
    budget = SharedRunBudget(config=test_config)
    scheduler = ParallelResearchScheduler(config=test_config)

    # b_1 depends on b_2, and b_2 depends on b_1
    briefs = [
        WorkerBrief(
            brief_id="b_1",
            title="B1",
            topic="Topic",
            objective="Obj",
            task_boundaries=["In", "Out"],
            depends_on=["b_2"],
        ),
        WorkerBrief(
            brief_id="b_2",
            title="B2",
            topic="Topic",
            objective="Obj",
            task_boundaries=["In", "Out"],
            depends_on=["b_1"],
        ),
    ]

    results = await scheduler.schedule_and_execute(briefs=briefs, shared_budget=budget)
    assert results["b_1"].status == WorkerStatus.FAILED
    assert results["b_2"].status == WorkerStatus.FAILED
    assert "Circular dependency" in (results["b_1"].error_message or "")


@pytest.mark.asyncio
async def test_scheduler_pre_exhausted_budget(test_config: ResearchTeamConfig) -> None:
    budget = SharedRunBudget(config=test_config, max_total_tokens=100, max_total_cost_usd=1.0)
    await budget.record_usage(tokens=150, cost_usd=0.01)

    scheduler = ParallelResearchScheduler(config=test_config)
    briefs = [
        WorkerBrief(
            brief_id="b_ex",
            title="Exhausted",
            topic="Topic",
            objective="Obj",
            task_boundaries=["In", "Out"],
        )
    ]

    results = await scheduler.schedule_and_execute(briefs=briefs, shared_budget=budget)
    assert results["b_ex"].status == WorkerStatus.FAILED
    assert "budget already exhausted" in (results["b_ex"].error_message or "").lower()
