"""Benchmark execution runner reporting Success Rate, Mean Steps, and Mean Cost."""

from __future__ import annotations

from typing import Any

from genai_core.evaluation import EvalRunner, ToolSelectionMetric
from genai_core.models.evaluation_model import EvalReport

from .dataset import create_15_goals_dataset
from .metrics import CostBudgetMetric, KeywordComplianceMetric, StepBudgetMetric


async def run_benchmark(
    target_agent_fn: Any,
    max_concurrency: int = 3,
) -> tuple[EvalReport, dict[str, float]]:
    """Execute the 15-goal benchmark suite and calculate summary KPIs."""
    dataset = create_15_goals_dataset()
    metrics = [
        ToolSelectionMetric(name="tool_selection"),
        KeywordComplianceMetric(),
        StepBudgetMetric(max_allowed_steps=8),
        CostBudgetMetric(max_allowed_cost=0.05),
    ]

    runner = EvalRunner(metrics=metrics, max_concurrency=max_concurrency)

    # Execute all 15 benchmark samples
    report: EvalReport = await runner.run(dataset=dataset, target=target_agent_fn)

    # Calculate run-level aggregations
    total_steps = 0
    total_cost = 0.0
    count = len(report.sample_results)

    for r in report.sample_results:
        steps = 1
        cost = 0.0
        if r.output is not None:
            if r.output.token_usage and "steps" in r.output.token_usage:
                steps = int(r.output.token_usage["steps"])
            if isinstance(r.output.raw_response, dict) and "cost_usd" in r.output.raw_response:
                cost = float(r.output.raw_response["cost_usd"])
        if cost == 0.0 and "cost_budget" in r.metrics and r.metrics["cost_budget"].details:
            cost = float(r.metrics["cost_budget"].details.get("cost_usd", 0.0))
        if steps == 1 and "step_budget" in r.metrics and r.metrics["step_budget"].details:
            steps = int(r.metrics["step_budget"].details.get("steps", 1))

        total_steps += steps
        total_cost += cost

    mean_steps = (total_steps / count) if count > 0 else 0.0
    mean_cost = (total_cost / count) if count > 0 else 0.0

    summary_stats = {
        "success_rate": report.pass_rate,
        "mean_steps": round(mean_steps, 2),
        "mean_cost_usd": round(mean_cost, 6),
        "total_cost_usd": round(total_cost, 6),
        "total_samples": count,
        "passed_samples": report.passed_samples,
    }

    return report, summary_stats


def format_benchmark_summary(stats: dict[str, Any]) -> str:
    """Format benchmark results into a clean, human-readable summary table."""
    return f"""
================================================================================
MODULE 17: BOUNDED RESEARCH & SUPPORT AGENT BENCHMARK REPORT
================================================================================
Total Goals Evaluated: {stats["total_samples"]}
Passed Samples:        {stats["passed_samples"]}
--------------------------------------------------------------------------------
Key Performance Indicators (KPIs):
  • Success Rate:      {stats["success_rate"]:.1%}
  • Mean Steps:        {stats["mean_steps"]} steps / goal
  • Mean Cost:         ${stats["mean_cost_usd"]:.6f} USD / goal
  • Total Run Cost:    ${stats["total_cost_usd"]:.6f} USD
================================================================================
"""


if __name__ == "__main__":
    # Test stub for direct CLI invocation
    print("Benchmark module initialized.")
