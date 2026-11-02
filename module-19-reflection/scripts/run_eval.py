#!/usr/bin/env python3
"""CLI benchmark runner for the 15-Topic Comparative Evaluation Suite."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project src to pythonpath
project_root = Path(__file__).resolve().parent.parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_runner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 15-topic comparative evaluation across Approaches A, B, and C."
    )
    parser.add_argument(
        "--topics",
        type=str,
        default="all",
        help="Comma-separated topic IDs to evaluate (e.g. 'qc-01,ai-01'), or 'all' for all 15 topics.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "benchmark_outputs"),
        help="Directory to save benchmark summary markdown and JSON data.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging output.",
    )
    return parser.parse_args()


async def main() -> int:
    from self_improving_report_writer.core.config import ReportWriterConfig
    from self_improving_report_writer.eval.benchmark_runner import BenchmarkRunner
    from self_improving_report_writer.eval.dataset import get_benchmark_dataset

    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    topic_ids = None
    if args.topics.strip().lower() != "all":
        topic_ids = [t.strip() for t in args.topics.split(",") if t.strip()]

    all_topics = get_benchmark_dataset()
    logger.info(f"Loaded benchmark dataset with {len(all_topics)} topics across 5 domains.")

    config = ReportWriterConfig()
    runner = BenchmarkRunner(config=config)

    logger.info("Starting benchmark evaluation...")
    report = await runner.run_benchmark(
        topic_ids=topic_ids,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 80)
    print(report.summary.to_markdown_table())
    print("=" * 80 + "\n")
    print(f"Results saved to: {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
