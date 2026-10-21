"""Integration tests executing the 15-goal benchmark evaluation suite."""

from __future__ import annotations

from typing import Any

import pytest
from genai_core.models.evaluation_model import EvalOutput, EvalSample

from research_and_support_agent.eval.runner import format_benchmark_summary, run_benchmark


@pytest.mark.asyncio
async def test_15_goals_benchmark_execution() -> None:
    """Verify that all 15 benchmark goals run and report success rate, mean steps, and mean cost."""

    # Deterministic mock target simulating the 15 scenarios
    async def mock_agent_target(sample: EvalSample) -> EvalOutput:
        expected_tool = sample.expected_tools[0] if sample.expected_tools else "ticket_status"
        tool_call_dict: list[dict[str, Any]] = [{"name": expected_tool, "arguments": {}}]

        # Simulate answers containing required keywords
        if sample.expected_answer and sample.expected_answer.required:
            answer_text = f"Output contains: {' '.join(sample.expected_answer.required)}."
        else:
            answer_text = "Task executed successfully."

        return EvalOutput(
            answer=answer_text,
            tool_calls=tool_call_dict,
            token_usage={"steps": 2, "input_tokens": 120, "output_tokens": 40},
            raw_response={"cost_usd": 0.00003},
        )

    _report, stats = await run_benchmark(target_agent_fn=mock_agent_target)

    # Assert benchmark coverage and performance KPIs
    assert stats["total_samples"] == 15
    assert stats["success_rate"] >= 0.85
    assert stats["mean_steps"] <= 5.0
    assert stats["mean_cost_usd"] > 0.0

    # Ensure format output prints cleanly
    summary_text = format_benchmark_summary(stats)
    assert "MODULE 17: BOUNDED RESEARCH & SUPPORT AGENT BENCHMARK REPORT" in summary_text
    assert "Success Rate:" in summary_text
    assert "Mean Steps:" in summary_text
    assert "Mean Cost:" in summary_text
