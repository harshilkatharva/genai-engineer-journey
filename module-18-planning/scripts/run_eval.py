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
from typing import Any

# Ensure local src directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dotenv import load_dotenv

from plan_and_execute_research_assistant.core import (
    get_assistant_config,
)
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
    config = get_assistant_config()

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
        default=None,
        help=f"Model identifier to use in live mode (fallback: {config.default_model})",
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
    # Parameters based on AssistantConfig (fallback to config values when not passed)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help=f"Maximum concurrent DAG tasks during execution (fallback: {config.max_concurrency})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=f"Global cumulative token budget ceiling (fallback: {config.max_total_tokens})",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help=f"Global cumulative cost budget in USD (fallback: ${config.max_total_cost_usd:.2f})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=f"Maximum wall-clock execution duration in seconds (fallback: {config.default_timeout_seconds}s)",
    )
    parser.add_argument(
        "--verification-threshold",
        type=float,
        default=None,
        help=f"Minimum confidence score for step verification (fallback: {config.verification_threshold})",
    )
    parser.add_argument(
        "--enable-episodic-similarity-search",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=f"Enable episodic memory similarity search (fallback: {config.enable_episodic_similarity_search})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=f"Sampling temperature for planning and generation (fallback: {config.default_temperature})",
    )
    parser.add_argument(
        "--max-replans",
        type=int,
        default=None,
        help=f"Maximum dynamic replanning cycles permitted per run (fallback: {config.max_replans})",
    )
    parser.add_argument(
        "--max-step-retries",
        type=int,
        default=None,
        help=f"Maximum execution retries for an individual step before replan (fallback: {config.max_step_retries})",
    )
    parser.add_argument(
        "--auto-approve-review",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=f"Whether to automatically approve plans during review (fallback: {config.auto_approve_review})",
    )
    parser.add_argument(
        "--pause-between-goals",
        type=float,
        default=None,
        help="Seconds to pause after each goal evaluation to prevent API rate limits (default: 30.0s in live mode, 0.0s in mock mode)",
    )
    args = parser.parse_args()

    # Resolve parameters: take from arguments if provided, else fall back to config / environment
    updates: dict[str, Any] = {}

    target_model = args.model or os.getenv("DEFAULT_LLM_MODEL") or config.default_model
    if target_model != config.default_model:
        updates["default_model"] = target_model

    db_url = (
        args.db_url
        or os.getenv("DATABASE_CONNECTION_CONVERSATION_URL")
        or "postgresql://postgres:postgres@127.0.0.1:5432/ai_search"
    )

    if args.max_concurrency is not None:
        updates["max_concurrency"] = args.max_concurrency
    if args.max_tokens is not None:
        updates["max_total_tokens"] = args.max_tokens
    if args.max_cost is not None:
        updates["max_total_cost_usd"] = args.max_cost
    if args.timeout is not None:
        updates["default_timeout_seconds"] = args.timeout
    if args.verification_threshold is not None:
        updates["verification_threshold"] = args.verification_threshold
    if args.enable_episodic_similarity_search is not None:
        updates["enable_episodic_similarity_search"] = args.enable_episodic_similarity_search
    if args.temperature is not None:
        updates["default_temperature"] = args.temperature
    if args.max_replans is not None:
        updates["max_replans"] = args.max_replans
    if args.max_step_retries is not None:
        updates["max_step_retries"] = args.max_step_retries
    if args.auto_approve_review is not None:
        updates["auto_approve_review"] = args.auto_approve_review

    effective_config = config.model_copy(update=updates) if updates else config

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
        f"LIVE API & POSTGRESQL [{effective_config.default_model}]"
        if is_live
        else "DETERMINISTIC MOCKS (OFFLINE/SAFE)"
    )
    print("\n================================================================================")
    print(f"MODULE 18: RUNNING COMPARATIVE BENCHMARK EVALUATION [Mode: {mode_label}]")
    print("================================================================================")
    print("ACTIVE CONFIGURATION:")
    print(f"  • Model:                  {effective_config.default_model}")
    print(f"  • Max Concurrency:        {effective_config.max_concurrency}")
    print(f"  • Max Tokens Budget:      {effective_config.max_total_tokens:,}")
    print(f"  • Max Cost Budget:        ${effective_config.max_total_cost_usd:.2f}")
    print(f"  • Execution Timeout:      {effective_config.default_timeout_seconds}s")
    print(f"  • Verification Threshold: {effective_config.verification_threshold}")
    print(f"  • Episodic Vector Search: {effective_config.enable_episodic_similarity_search}")
    print(f"  • Dynamic Replans Cap:    {effective_config.max_replans}")
    if is_live:
        print(f"  • Database URL:           {db_url}")
    print("--------------------------------------------------------------------------------\n")

    samples = get_10_benchmark_goals()
    if args.goal_ids:
        target_ids = {gid.strip() for gid in args.goal_ids.split(",") if gid.strip()}
        samples = [s for s in samples if str(s.id) in target_ids]
    if args.limit:
        samples = samples[: args.limit]

    if not is_live:
        # Offline / deterministic mock benchmark execution
        print(f"Evaluating {len(samples)} curated benchmark goals under deterministic mocks...\n")

        pause_sec = args.pause_between_goals if args.pause_between_goals is not None else 0.0
        report = await run_comparative_benchmark(
            samples=samples,
            planned_target=deterministic_mock_planned_target,
            reactive_target=deterministic_mock_reactive_target,
            max_concurrency=effective_config.max_concurrency,
            pause_between_goals_seconds=pause_sec,
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
        f"Connecting to live services and executing benchmark (Planned vs. Reactive) using model '{effective_config.default_model}'..."
    )
    planned_target, reactive_target = await create_live_benchmark_targets(
        api_key=api_key,
        db_url=db_url,
        model=effective_config.default_model,
        config=effective_config,
    )
    print(f"Evaluating {len(samples)} goals with live Gemini LLM and PostgreSQL...\n")
    live_pause_sec = args.pause_between_goals if args.pause_between_goals is not None else 30.0
    report = await run_comparative_benchmark(
        samples=samples,
        planned_target=planned_target,
        reactive_target=reactive_target,
        max_concurrency=args.max_concurrency if args.max_concurrency is not None else 1,
        pause_between_goals_seconds=live_pause_sec,
    )
    print(report.format_comparison_table())
    print("\nLive benchmark completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
