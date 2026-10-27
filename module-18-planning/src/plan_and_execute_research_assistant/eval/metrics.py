"""Metrics and comparative report generation for Module 18 evaluation benchmark."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SystemEvalResult(BaseModel):
    """Execution telemetry and evaluation outcome for a single goal under a specific system."""

    system_name: str = Field(description="Name of the system evaluated (Planned or Reactive)")
    goal_id: str = Field(description="Benchmark goal identifier (e.g. 'G-01')")
    category: str = Field(description="Goal category (research, operations, resilience, mixed)")
    passed: bool = Field(description="Whether the system output satisfied evaluation criteria")
    steps: int = Field(default=1, ge=0, description="Total turns / execution steps taken")
    tokens: int = Field(default=0, ge=0, description="Total tokens consumed")
    cost_usd: float = Field(default=0.0, ge=0.0, description="Total dollar cost incurred in USD")
    latency_s: float = Field(default=0.0, ge=0.0, description="Wall-clock duration in seconds")
    replan_count: int = Field(default=0, ge=0, description="Number of replans executed")
    deliverable: str = Field(default="", description="Final synthesized deliverable")
    error_message: str | None = Field(default=None, description="Error detail if execution failed")


class BenchmarkSummaryStats(BaseModel):
    """Aggregated key performance indicators across all benchmark goals for one system."""

    system_name: str = Field(description="Name of the evaluated system")
    total_goals: int = Field(default=0, ge=0, description="Total goals evaluated")
    passed_goals: int = Field(default=0, ge=0, description="Number of goals passed successfully")
    completion_rate: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Completion percentage (0.0 to 100.0%)"
    )
    mean_steps: float = Field(default=0.0, ge=0.0, description="Mean steps / turns per goal")
    mean_tokens: float = Field(default=0.0, ge=0.0, description="Mean tokens consumed per goal")
    mean_cost_usd: float = Field(default=0.0, ge=0.0, description="Mean cost in USD per goal")
    mean_latency_s: float = Field(
        default=0.0, ge=0.0, description="Mean wall-clock latency in seconds per goal"
    )
    total_cost_usd: float = Field(default=0.0, ge=0.0, description="Cumulative run cost in USD")
    total_latency_s: float = Field(
        default=0.0, ge=0.0, description="Cumulative wall-clock time in seconds"
    )
    replan_recovery_rate: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Percentage of replanned runs that succeeded"
    )

    @classmethod
    def compute(cls, system_name: str, results: list[SystemEvalResult]) -> BenchmarkSummaryStats:
        """Calculate summary statistics from a collection of goal results."""
        total = len(results)
        if total == 0:
            return cls(system_name=system_name)

        passed = sum(1 for r in results if r.passed)
        total_steps = sum(r.steps for r in results)
        total_tokens = sum(r.tokens for r in results)
        total_cost = sum(r.cost_usd for r in results)
        total_latency = sum(r.latency_s for r in results)

        # Replan recovery rate: percentage of goals that triggered replans and still passed
        replanned_runs = [r for r in results if r.replan_count > 0]
        if replanned_runs:
            recovered_count = sum(1 for r in replanned_runs if r.passed)
            replan_recovery = round((recovered_count / len(replanned_runs)) * 100.0, 1)
        else:
            replan_recovery = 100.0 if passed == total else 0.0

        return cls(
            system_name=system_name,
            total_goals=total,
            passed_goals=passed,
            completion_rate=round((passed / total) * 100.0, 1),
            mean_steps=round(total_steps / total, 2),
            mean_tokens=round(total_tokens / total, 1),
            mean_cost_usd=round(total_cost / total, 6),
            mean_latency_s=round(total_latency / total, 3),
            total_cost_usd=round(total_cost, 6),
            total_latency_s=round(total_latency, 3),
            replan_recovery_rate=replan_recovery,
        )


class ComparativeBenchmarkReport(BaseModel):
    """Comprehensive side-by-side comparative report evaluating Planned vs Reactive architectures."""

    planned_stats: BenchmarkSummaryStats
    reactive_stats: BenchmarkSummaryStats
    planned_results: list[SystemEvalResult]
    reactive_results: list[SystemEvalResult]

    def format_comparison_table(self) -> str:
        """Format benchmark findings into a clean, human-readable comparative table."""
        lines: list[str] = [
            "=" * 92,
            "MODULE 18 vs MODULE 17: FAIR COMPARATIVE BENCHMARK EVALUATION REPORT",
            "=" * 92,
            f"{'Goal ID':<8} {'Category':<12} {'Planned Status':<16} {'Planned (Steps/Lat/Cost)':<26} {'Reactive Status':<16} {'Reactive (Steps/Lat/Cost)':<26}",
            "-" * 92,
        ]

        # Map results by goal id
        planned_map = {r.goal_id: r for r in self.planned_results}
        reactive_map = {r.goal_id: r for r in self.reactive_results}
        all_ids = sorted(set(planned_map.keys()) | set(reactive_map.keys()))

        for g_id in all_ids:
            p_res = planned_map.get(g_id)
            r_res = reactive_map.get(g_id)

            category = p_res.category if p_res else (r_res.category if r_res else "unknown")
            p_status = "PASSED" if (p_res and p_res.passed) else "FAILED"
            r_status = "PASSED" if (r_res and r_res.passed) else "FAILED"

            p_detail = (
                f"{p_res.steps} stp | {p_res.latency_s:.2f}s | ${p_res.cost_usd:.5f}"
                if p_res
                else "N/A"
            )
            r_detail = (
                f"{r_res.steps} stp | {r_res.latency_s:.2f}s | ${r_res.cost_usd:.5f}"
                if r_res
                else "N/A"
            )

            lines.append(
                f"{g_id:<8} {category:<12} {p_status:<16} {p_detail:<26} {r_status:<16} {r_detail:<26}"
            )

        p = self.planned_stats
        r = self.reactive_stats

        # Compute architectural advantages
        latency_diff = r.mean_latency_s - p.mean_latency_s
        latency_win = (
            f"{((latency_diff / r.mean_latency_s) * 100):.1f}% faster"
            if r.mean_latency_s > 0 and latency_diff > 0
            else f"{abs(latency_diff):.2f}s difference"
        )
        completion_diff = p.completion_rate - r.completion_rate

        lines.extend(
            [
                "-" * 92,
                "HEAD-TO-HEAD KPI SUMMARY:",
                f"  • Completion Rate:        Planned {p.completion_rate:>5.1f}%   vs   Reactive {r.completion_rate:>5.1f}%  (Δ {completion_diff:+.1f}%)",
                f"  • Mean Steps / Goal:      Planned {p.mean_steps:>5.2f}     vs   Reactive {r.mean_steps:>5.2f}  (Δ {p.mean_steps - r.mean_steps:+.2f})",
                f"  • Mean Latency / Goal:    Planned {p.mean_latency_s:>5.3f}s    vs   Reactive {r.mean_latency_s:>5.3f}s  ({latency_win})",
                f"  • Mean Cost / Goal:       Planned ${p.mean_cost_usd:>8.6f} vs   Reactive ${r.mean_cost_usd:>8.6f}",
                f"  • Total Evaluation Cost:  Planned ${p.total_cost_usd:>8.6f} vs   Reactive ${r.total_cost_usd:>8.6f}",
                f"  • Replan Recovery Rate:   Planned {p.replan_recovery_rate:>5.1f}%   vs   Reactive N/A (single loop)",
                "=" * 92,
            ]
        )

        return "\n".join(lines)
