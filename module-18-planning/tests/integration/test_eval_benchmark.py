"""Integration tests for the Module 18 10-Goal Comparative Evaluation Benchmark."""

from __future__ import annotations

import pytest

from plan_and_execute_research_assistant.eval import (
    BENCHMARK_GOALS,
    BenchmarkSummaryStats,
    ComparativeBenchmarkReport,
    SystemEvalResult,
    create_10_goals_dataset,
    deterministic_mock_planned_target,
    deterministic_mock_reactive_target,
    get_10_benchmark_goals,
    run_comparative_benchmark,
)


def test_10_benchmark_goals_structure() -> None:
    """Verify that all 10 curated benchmark goals satisfy structural constraints."""
    goals = get_10_benchmark_goals()
    assert len(goals) == 10
    assert len(BENCHMARK_GOALS) == 10

    # Ensure all IDs are unique and sequentially formatted G-01 through G-10
    ids = [str(g.id) for g in goals]
    assert len(set(ids)) == 10
    expected_ids = [f"G-{i:02d}" for i in range(1, 11)]
    assert ids == expected_ids

    # Validate mandatory fields across all goals
    for g in goals:
        assert g.query and len(g.query.strip()) > 10
        assert g.expected_tools and len(g.expected_tools) >= 1
        assert g.expected_answer is not None
        assert len(g.expected_answer.required) >= 1
        assert g.metadata and "category" in g.metadata
        assert g.metadata["category"] in {"research", "operations", "resilience", "mixed"}


def test_create_10_goals_dataset() -> None:
    """Verify EvalDataset wrapper construction."""
    ds = create_10_goals_dataset()
    assert ds.metadata["name"] == "module_18_10_goals_benchmark"
    assert len(ds.samples) == 10
    assert ds.metadata["module"] == 18


@pytest.mark.asyncio
async def test_run_comparative_benchmark_mock_mode() -> None:
    """Verify end-to-end execution of the comparative benchmark under deterministic mocks."""
    report: ComparativeBenchmarkReport = await run_comparative_benchmark(
        planned_target=deterministic_mock_planned_target,
        reactive_target=deterministic_mock_reactive_target,
        max_concurrency=4,
        pause_between_goals_seconds=0.0,
    )

    # 1. Planned Assistant assertions
    p_stats = report.planned_stats
    assert p_stats.total_goals == 10
    assert p_stats.passed_goals == 10
    assert p_stats.completion_rate == 100.0
    assert p_stats.replan_recovery_rate == 100.0
    assert p_stats.mean_steps > 0
    assert p_stats.mean_latency_s > 0

    # 2. Reactive Baseline assertions
    r_stats = report.reactive_stats
    assert r_stats.total_goals == 10
    assert r_stats.passed_goals == 9
    assert r_stats.completion_rate == 90.0

    # 3. Architectural comparison assertions
    # Planned should have lower mean latency due to concurrent branches
    assert p_stats.mean_latency_s < r_stats.mean_latency_s
    # Planned should have fewer mean steps than reactive multi-turn loop
    assert p_stats.mean_steps < r_stats.mean_steps

    # 4. G-06 Resilience goal verification
    p_g06 = next(r for r in report.planned_results if r.goal_id == "G-06")
    r_g06 = next(r for r in report.reactive_results if r.goal_id == "G-06")
    assert p_g06.passed is True
    assert p_g06.replan_count == 1
    assert r_g06.passed is False
    assert r_g06.replan_count == 0

    # 5. Formatted table output
    table = report.format_comparison_table()
    assert "MODULE 18 vs MODULE 17" in table
    assert "G-01" in table
    assert "G-06" in table
    assert "G-10" in table
    assert "Completion Rate:" in table
    assert "Replan Recovery Rate:" in table


def test_benchmark_metrics_computation() -> None:
    """Verify BenchmarkSummaryStats mathematical calculations."""
    results = [
        SystemEvalResult(
            system_name="TestSystem",
            goal_id="G-01",
            category="research",
            passed=True,
            steps=3,
            tokens=300,
            cost_usd=0.03,
            latency_s=1.0,
            replan_count=0,
        ),
        SystemEvalResult(
            system_name="TestSystem",
            goal_id="G-02",
            category="resilience",
            passed=False,
            steps=5,
            tokens=500,
            cost_usd=0.05,
            latency_s=2.0,
            replan_count=1,
        ),
    ]

    stats = BenchmarkSummaryStats.compute(system_name="TestSystem", results=results)
    assert stats.total_goals == 2
    assert stats.passed_goals == 1
    assert stats.completion_rate == 50.0
    assert stats.mean_steps == 4.0
    assert stats.mean_tokens == 400.0
    assert stats.mean_cost_usd == pytest.approx(0.04)
    assert stats.mean_latency_s == pytest.approx(1.5)
    assert stats.replan_recovery_rate == 0.0  # 1 replanned run, 0 passed
