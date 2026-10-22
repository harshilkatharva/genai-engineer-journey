"""Unit tests for the evaluation runner and summary formatter."""

from __future__ import annotations

from research_and_support_agent.eval.runner import format_benchmark_summary


def test_format_benchmark_summary() -> None:
    """Verify human-readable benchmark summary text formatting."""
    stats = {
        "success_rate": 1.0,
        "mean_steps": 1.4,
        "mean_cost_usd": 0.000045,
        "total_cost_usd": 0.000675,
        "total_samples": 15,
        "passed_samples": 15,
    }
    summary_text = format_benchmark_summary(stats)
    assert "BENCHMARK REPORT" in summary_text
    assert "100.0%" in summary_text
    assert "1.4 steps / goal" in summary_text
    assert "$0.000045 USD / goal" in summary_text
    assert "Total Goals Evaluated: 15" in summary_text
