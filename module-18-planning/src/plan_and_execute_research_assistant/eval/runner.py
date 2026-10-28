"""Runner executing comparative evaluation across Planned vs. Reactive architectures."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from .dataset import get_10_benchmark_goals
from .metrics import (
    BenchmarkSummaryStats,
    ComparativeBenchmarkReport,
    SystemEvalResult,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from genai_core.models.evaluation_model import EvalSample


async def deterministic_mock_planned_target(sample: EvalSample) -> SystemEvalResult:
    """Simulate execution of Module 18 Plan-and-Execute Assistant under deterministic offline mocks.

    Simulates:
    - Parallel branch dispatch (e.g. 2-3 parallel steps merging into 1 synthesis step).
    - Reduced wall-clock latency due to concurrency.
    - Concurrency-safe replanning on resilience goal G-06 (recovering successfully with replan_count=1).
    - Reliable criteria satisfaction via two-tier verification.
    """
    g_id = str(sample.id)
    category = sample.metadata.get("category", "general")
    required = sample.expected_answer.required if sample.expected_answer else []

    # Simulate realistic steps, tokens, and parallel speedups
    if g_id == "G-06":
        # Resilience test: step failed, replanning engine triggered, successfully recovered
        replan_count = 1
        steps = 4
        tokens = 720
        cost = 0.000072
        latency_s = 0.075
        deliverable = f"Replanning recovered successfully. Retrieved required documentation: {' '.join(required)}."
        passed = True
    elif category == "research":
        # Parallel research branches
        replan_count = 0
        steps = 3
        tokens = 580
        cost = 0.000058
        latency_s = 0.045
        deliverable = (
            f"Synthesized research deliverable covering: {' '.join(required)}. "
            f"Reference: {sample.expected_answer.reference if sample.expected_answer else ''}"
        )
        passed = True
    elif g_id == "G-07":
        # 3 parallel ticket fetches + 1 synthesis
        replan_count = 0
        steps = 4
        tokens = 620
        cost = 0.000062
        latency_s = 0.048
        deliverable = f"Consolidated executive brief covering all tickets: {' '.join(required)}."
        passed = True
    else:
        # Operations or Mixed
        replan_count = 0
        steps = 3
        tokens = 510
        cost = 0.000051
        latency_s = 0.052
        deliverable = f"Operation completed successfully with verified steps: {' '.join(required)}."
        passed = True

    return SystemEvalResult(
        system_name="Module 18 Planned Assistant",
        goal_id=g_id,
        category=category,
        passed=passed,
        steps=steps,
        tokens=tokens,
        cost_usd=cost,
        latency_s=latency_s,
        replan_count=replan_count,
        deliverable=deliverable,
    )


async def deterministic_mock_reactive_target(sample: EvalSample) -> SystemEvalResult:
    """Simulate execution of Module 17 Reactive BoundedAgent under deterministic offline mocks.

    Simulates:
    - Sequential single-agent ReAct loop (higher turns and latency).
    - Lack of dynamic replanning: fails on ambiguous/resilience goal G-06 due to turn exhaustion.
    - Higher cumulative tokens due to appending full conversational history on each turn.
    """
    g_id = str(sample.id)
    category = sample.metadata.get("category", "general")
    required = sample.expected_answer.required if sample.expected_answer else []

    if g_id == "G-06":
        # Reactive baseline gets stuck in ambiguous loop and exceeds max steps without replanner
        replan_count = 0
        steps = 6  # max steps reached
        tokens = 1150
        cost = 0.000115
        latency_s = 0.160
        deliverable = "Agent turn limit exceeded without resolving ambiguous query."
        passed = False
        error_message = "Turn limit reached (ToolLoopStatus.MAX_STEPS_REACHED)"
    elif category == "research":
        # Sequential tool calls take significantly longer wall-clock time
        replan_count = 0
        steps = 5
        tokens = 920
        cost = 0.000092
        latency_s = 0.125
        deliverable = f"Reactive research findings covering: {' '.join(required)}."
        passed = True
        error_message = None
    elif g_id == "G-07":
        # 3 tickets fetched sequentially instead of concurrently
        replan_count = 0
        steps = 6
        tokens = 980
        cost = 0.000098
        latency_s = 0.142
        deliverable = f"Sequential ticket summary for {' '.join(required)}."
        passed = True
        error_message = None
    else:
        replan_count = 0
        steps = 4
        tokens = 780
        cost = 0.000078
        latency_s = 0.105
        deliverable = f"Reactive operation completed for {' '.join(required)}."
        passed = True
        error_message = None

    return SystemEvalResult(
        system_name="Module 17 Reactive BoundedAgent",
        goal_id=g_id,
        category=category,
        passed=passed,
        steps=steps,
        tokens=tokens,
        cost_usd=cost,
        latency_s=latency_s,
        replan_count=replan_count,
        deliverable=deliverable,
        error_message=error_message,
    )


async def run_comparative_benchmark(
    samples: list[EvalSample] | None = None,
    planned_target: (
        Callable[[EvalSample], Awaitable[SystemEvalResult]] | None
    ) = deterministic_mock_planned_target,
    reactive_target: (
        Callable[[EvalSample], Awaitable[SystemEvalResult]] | None
    ) = deterministic_mock_reactive_target,
    max_concurrency: int = 3,
    pause_between_goals_seconds: float = 30.0,
) -> ComparativeBenchmarkReport:
    """Execute head-to-head evaluation benchmark comparing Planned and Reactive systems.

    Args:
        samples: Benchmark goals to evaluate (defaults to the 10 curated goals).
        planned_target: Target runner for Module 18 Planned Assistant.
        reactive_target: Target runner for Module 17 Reactive BoundedAgent.
        max_concurrency: Maximum parallel goal evaluations.
        pause_between_goals_seconds: Cooldown duration in seconds to pause after each goal evaluation.

    Returns:
        ComparativeBenchmarkReport containing individual results and side-by-side KPI analysis.
    """
    eval_samples = samples or get_10_benchmark_goals()
    sem = asyncio.Semaphore(max_concurrency)

    async def _eval_sample(
        sample: EvalSample,
        target_fn: Callable[[EvalSample], Awaitable[SystemEvalResult]],
    ) -> SystemEvalResult:
        async with sem:
            start_t = time.monotonic()
            try:
                res = await target_fn(sample)
                # Ensure latency stamped if not populated by mock
                if res.latency_s <= 0.0:
                    res.latency_s = round(time.monotonic() - start_t, 3)
            except Exception as exc:  # noqa: BLE001
                elapsed = round(time.monotonic() - start_t, 3)
                res = SystemEvalResult(
                    system_name=getattr(target_fn, "__name__", "Target"),
                    goal_id=str(sample.id),
                    category=sample.metadata.get("category", "unknown"),
                    passed=False,
                    steps=1,
                    tokens=0,
                    cost_usd=0.0,
                    latency_s=elapsed,
                    replan_count=0,
                    deliverable="",
                    error_message=str(exc),
                )

            # Pause after each goal evaluation to prevent LLM API rate limits and quotas
            if pause_between_goals_seconds > 0.0:
                print(
                    f"  [Cooldown] Pausing {pause_between_goals_seconds:.0f}s after goal {sample.id} ({res.system_name})...",
                    flush=True,
                )
                await asyncio.sleep(pause_between_goals_seconds)

            return res

    # Execute Planned Assistant across all goals
    planned_fn = planned_target or deterministic_mock_planned_target
    planned_tasks = [_eval_sample(s, planned_fn) for s in eval_samples]
    planned_results: list[SystemEvalResult] = list(await asyncio.gather(*planned_tasks))

    # Execute Reactive Baseline across all goals
    reactive_fn = reactive_target or deterministic_mock_reactive_target
    reactive_tasks = [_eval_sample(s, reactive_fn) for s in eval_samples]
    reactive_results: list[SystemEvalResult] = list(await asyncio.gather(*reactive_tasks))

    planned_stats = BenchmarkSummaryStats.compute(
        system_name="Module 18 Planned Assistant",
        results=planned_results,
    )
    reactive_stats = BenchmarkSummaryStats.compute(
        system_name="Module 17 Reactive BoundedAgent",
        results=reactive_results,
    )

    return ComparativeBenchmarkReport(
        planned_stats=planned_stats,
        reactive_stats=reactive_stats,
        planned_results=planned_results,
        reactive_results=reactive_results,
    )
