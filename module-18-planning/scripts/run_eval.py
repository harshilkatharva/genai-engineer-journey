#!/usr/bin/env python3
"""Module 18: Runnable Fair Comparative Evaluation Benchmark Script.

Executes the 10-goal benchmark suite comparing:
- Candidate: Module 18 Plan-and-Execute Research Assistant (DAG planning, parallel execution, dynamic replanning)
- Baseline: Module 17 Reactive BoundedAgent (Sequential ReAct loop)

Supports two execution modes:
1. Deterministic mocks (default / offline mode): safe for CI/CD, fast verification, zero token/cost charges.
2. Live environment (--live mode): connects to live LLM (Google Gemini 3.5 Flash Lite),
   PostgreSQL database, and web search under identical resource caps (40k tokens, $0.25 USD).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Ensure local src directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dotenv import load_dotenv

from plan_and_execute_research_assistant.eval import (
    create_live_benchmark_targets,
    deterministic_mock_planned_target,
    deterministic_mock_reactive_target,
    get_10_benchmark_goals,
    run_comparative_benchmark,
)


async def main() -> None:
    """Entrypoint for the comparative evaluation benchmark runner."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run the Module 18 10-Goal Planned vs Reactive Comparative Benchmark"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run against live LLM API, PostgreSQL database, and web search.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run against deterministic offline mocks (safe for CI/CD).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("DEFAULT_LLM_MODEL", "gemini-3.5-flash-lite"),
        help="Model identifier to use in live mode (default: gemini-3.5-flash-lite)",
    )
    parser.add_argument(
        "--db-url",
        default=os.getenv(
            "DATABASE_CONNECTION_CONVERSATION_URL",
            "postgresql://postgres:postgres@127.0.0.1:5432/ai_search",
        ),
        help="PostgreSQL connection URL for live mode",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit evaluation to the first N benchmark goals.",
    )
    parser.add_argument(
        "--goal-ids",
        default=None,
        help="Comma-separated list of goal IDs to evaluate (e.g. G-01,G-02).",
    )
    args = parser.parse_args()

    # Determine execution mode: CLI flags or interactive prompt
    is_live = False
    if args.live:
        is_live = True
    elif args.mock:
        is_live = False
    else:
        print("\n" + "=" * 80)
        print("MODULE 18: PLAN-AND-EXECUTE RESEARCH ASSISTANT BENCHMARK EVALUATION")
        print("=" * 80)
        print("Select evaluation option:")
        print("1. deterministic_mock eval")
        print("2. realistic eval run")
        print("-" * 80)
        try:
            choice = input("Enter choice (1 or 2): ").strip()
        except (EOFError, KeyboardInterrupt):
            choice = "1"
        is_live = choice in ("2", "realistic", "realistic eval run", "2. realistic eval run")

    mode_label = (
        f"LIVE API & POSTGRESQL [{args.model}]" if is_live else "DETERMINISTIC MOCKS (OFFLINE/SAFE)"
    )
    print("\n================================================================================")
    print(f"MODULE 18: RUNNING COMPARATIVE BENCHMARK EVALUATION [Mode: {mode_label}]")
    print("================================================================================")

    samples = get_10_benchmark_goals()
    if args.goal_ids:
        target_ids = {gid.strip() for gid in args.goal_ids.split(",") if gid.strip()}
        samples = [s for s in samples if str(s.id) in target_ids]
    if args.limit:
        samples = samples[: args.limit]

    if not is_live:
        # Offline / deterministic mock benchmark execution
        print(f"Evaluating {len(samples)} curated benchmark goals under deterministic mocks...\n")

        report = await run_comparative_benchmark(
            samples=samples,
            planned_target=deterministic_mock_planned_target,
            reactive_target=deterministic_mock_reactive_target,
            max_concurrency=3,
        )

        print(report.format_comparison_table())
        print("\nBenchmark completed successfully.")
        return

    # Live execution mode
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(
            "ERROR: GOOGLE_API_KEY or GEMINI_API_KEY environment variable is required for live evaluation."
        )
        sys.exit(1)

    print(
        f"Connecting to live services and executing benchmark (Planned vs. Reactive) using model '{args.model}'..."
    )
    planned_target, reactive_target = await create_live_benchmark_targets(
        api_key=api_key,
        db_url=args.db_url,
        model=args.model,
    )
    print(f"Evaluating {len(samples)} goals with live Gemini LLM and PostgreSQL...\n")
    report = await run_comparative_benchmark(
        samples=samples,
        planned_target=planned_target,
        reactive_target=reactive_target,
        max_concurrency=1,
    )
    print(report.format_comparison_table())
    print("\nLive benchmark completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
