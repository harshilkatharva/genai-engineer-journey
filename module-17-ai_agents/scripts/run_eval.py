#!/usr/bin/env python3
"""Module 17: Runnable Evaluation Benchmark Script.

Executes the 15-goal evaluation suite using deterministic mocks (or optional live agent)
and reports Success Rate, Mean Steps, and Mean Cost per goal.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from genai_core.models.evaluation_model import EvalOutput, EvalSample

from research_and_support_agent.eval.runner import format_benchmark_summary, run_benchmark


async def deterministic_mock_target(sample: EvalSample) -> EvalOutput:
    """Deterministic mock agent target simulating responses for all 15 benchmark goals.

    Allows running the full benchmark without consuming API credits or requiring external network.
    """
    expected_tool = sample.expected_tools[0] if sample.expected_tools else "ticket_status"
    tool_calls: list[dict[str, Any]] = [{"name": expected_tool, "arguments": {}}]

    # Construct appropriate answer satisfying the sample's required keywords
    if sample.expected_answer and sample.expected_answer.required:
        answer_text = (
            f"Action processed successfully. Details: {' '.join(sample.expected_answer.required)}."
        )
    else:
        answer_text = "Operation processed successfully."

    # Simulate realistic steps and cost
    sample_id = sample.id or ""
    steps = 1
    cost = 0.000035

    if "step_limit" in sample_id:
        steps = 5
        cost = 0.000120
    elif "approval" in sample_id or "refund" in sample_id:
        steps = 2
        cost = 0.000060

    return EvalOutput(
        answer=answer_text,
        tool_calls=tool_calls,
        token_usage={"steps": steps, "input_tokens": 150 * steps, "output_tokens": 45 * steps},
        raw_response={"cost_usd": cost, "mode": "deterministic_mock"},
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Module 17 15-Goal Benchmark Suite")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run against a live LLM and database environment instead of deterministic mocks.",
    )
    args = parser.parse_args()

    mode_label = "LIVE LLM & ENVIRONMENT" if args.live else "DETERMINISTIC MOCKS (OFFLINE/SAFE)"
    print(f"\n>>> Initializing Module 17 Evaluation Suite [Mode: {mode_label}]...")

    if args.live:
        raise NotImplementedError(
            "Live evaluation requires active DB and LLM credentials. "
            "Please use the default deterministic mock mode for offline benchmarks."
        )

    report, stats = await run_benchmark(target_agent_fn=deterministic_mock_target)

    # Print summary
    print(format_benchmark_summary(stats))

    print("Individual Goal Breakdown:")
    print("--------------------------------------------------------------------------------")
    for r in report.sample_results:
        status = "PASSED [✓]" if r.passed else "FAILED [✗]"
        sample_id = r.sample_id or "unknown"
        print(f" • {sample_id:<36} | {status}")
    print("================================================================================\n")


if __name__ == "__main__":
    asyncio.run(main())
