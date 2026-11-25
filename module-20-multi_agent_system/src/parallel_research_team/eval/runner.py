"""End-to-end evaluation runner comparing Single Agent vs Multi-Agent across benchmark dataset."""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from genai_core import DocumentChunk, InMemoryVectorStore, RetrievalService
from genai_core.embedding.local_provider import LocalEmbeddingProvider
from genai_core.observability import get_logger
from genai_core.services.embedding_service import EmbeddingService

from ..core.config import ResearchTeamConfig, get_research_team_config
from ..workers.tools import create_retrieval_service
from .dataset import (
    get_15_benchmark_questions,
    get_benchmark_question_by_id,
)
from .judge import EvaluationJudge
from .models import (
    EvaluationQuestion,
    EvaluationRunSummary,
    ExpectedWinner,
    QuestionEvaluationResult,
)
from .multi_agent_adapter import MultiAgentAdapter
from .rate_limiter import AsyncRateLimiter, get_global_rate_limiter
from .single_agent_adapter import SingleAgentAdapter

load_dotenv()
_logger = get_logger("parallel_research_team.eval.runner")

DEFAULT_EVAL_RESULTS_DIR = Path("eval_results")
DEFAULT_FIXTURES_PATH = Path("tests/fixtures/research_chunks.json")


def setup_eval_retrieval_service(
    config: ResearchTeamConfig, fixtures_path: Path = DEFAULT_FIXTURES_PATH
) -> RetrievalService | None:
    """Set up retrieval service respecting config.retrieval_mode."""
    try:
        if config.retrieval_mode == "postgres" and config.postgres_connection_string:
            emb_provider = LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
            emb_service = EmbeddingService(provider=emb_provider)
            return create_retrieval_service(config=config, embedding_service=emb_service)

        if fixtures_path.exists():
            content = fixtures_path.read_text(encoding="utf-8")
            raw_chunks: list[dict[str, Any]] = json.loads(content)[:100]

            document_chunks = [
                DocumentChunk(
                    id=c["chunk_id"],
                    content=c["chunk_text"],
                    embedding=json.loads(c["embedding"])
                    if isinstance(c["embedding"], str)
                    else c.get("embedding", []),
                    metadata=c.get("metadata") or {},
                    source_id=c.get("metadata", {}).get("document_name", "production_handbook"),
                )
                for c in raw_chunks
            ]
            store = InMemoryVectorStore(dimensions=384, chunks=document_chunks)
            emb_provider = LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
            emb_service = EmbeddingService(provider=emb_provider)
            return RetrievalService(store=store, embedding_service=emb_service)

        return create_retrieval_service(config=config)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("Could not initialize retrieval service: %s. Proceeding without KB.", exc)
        return None


class EvaluationPipelineRunner:
    """Orchestrates comparative benchmarking between Module 17 Single Agent and Module 20 Multi-Agent."""

    def __init__(
        self,
        config: ResearchTeamConfig | None = None,
        retrieval_service: RetrievalService | None = None,
        rate_limiter: AsyncRateLimiter | None = None,
        inter_question_cooldown_seconds: float = 30.0,
        output_dir: Path | None = None,
    ) -> None:
        self.config = config or get_research_team_config()
        self.retrieval_service = retrieval_service
        self.rate_limiter = rate_limiter or get_global_rate_limiter()
        self.cooldown_seconds = max(0.0, inter_question_cooldown_seconds)
        self.output_dir = output_dir or DEFAULT_EVAL_RESULTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.single_agent = SingleAgentAdapter(
            config=self.config,
            retrieval_service=self.retrieval_service,
            rate_limiter=self.rate_limiter,
        )
        self.multi_agent = MultiAgentAdapter(
            config=self.config,
            retrieval_service=self.retrieval_service,
            rate_limiter=self.rate_limiter,
        )
        self.judge = EvaluationJudge(
            config=self.config,
            rate_limiter=self.rate_limiter,
        )

    async def evaluate_single_question(
        self,
        question: EvaluationQuestion,
    ) -> QuestionEvaluationResult:
        """Execute a single benchmark question across both systems and evaluate via judge."""
        print("\n" + "=" * 80)
        print(f"EVALUATING QUESTION: [{question.id}] ({question.category.value})")
        print(f"Query: '{question.query}'")
        print(f"Expected Winner: {question.expected_winner.value}")
        print("=" * 80)

        # 1. Run Single Agent (Module 17)
        print("\n--> [1/2] Running Module 17 BoundedAgent (Single-Agent Baseline)...")
        single_answer, single_metrics = await self.single_agent.run(question.query)
        print(
            f"✔ Single Agent completed in {single_metrics.latency_seconds:.2f}s | "
            f"{single_metrics.tokens:,} tokens | ${single_metrics.cost_usd:.5f} | "
            f"Status: {single_metrics.status}"
        )

        # Inter-agent spacing (10s) to protect RPM limits
        print("[Rate-limit safety] Pausing 10s between agents...")
        await asyncio.sleep(10.0)

        # 2. Run Multi-Agent System (Module 20)
        print("\n--> [2/2] Running Module 20 ParallelResearchTeam (Multi-Agent System)...")
        multi_answer, multi_metrics = await self.multi_agent.run(question.query)
        print(
            f"✔ Multi-Agent completed in {multi_metrics.latency_seconds:.2f}s | "
            f"{multi_metrics.tokens:,} tokens | ${multi_metrics.cost_usd:.5f} | "
            f"Status: {multi_metrics.status}"
        )

        # 3. LLM-as-a-Judge Evaluation
        print("\n--> [3/3] Running LLM-as-a-Judge Quality Scoring...")
        single_quality = await self.judge.evaluate_answer(
            question=question,
            answer=single_answer,
            system_name="Module 17 Single Agent",
        )
        print(
            f"  • Single Agent Quality: {single_quality.overall_score:.2f}/5.0 "
            f"(Fact: {single_quality.factuality:.1f}, Comp: {single_quality.comprehensiveness:.1f}, "
            f"Clar: {single_quality.clarity:.1f}, Rel: {single_quality.relevance:.1f})"
        )

        multi_quality = await self.judge.evaluate_answer(
            question=question,
            answer=multi_answer,
            system_name="Module 20 Multi-Agent",
        )
        print(
            f"  • Multi-Agent Quality:  {multi_quality.overall_score:.2f}/5.0 "
            f"(Fact: {multi_quality.factuality:.1f}, Comp: {multi_quality.comprehensiveness:.1f}, "
            f"Clar: {multi_quality.clarity:.1f}, Rel: {multi_quality.relevance:.1f})"
        )

        # 4. Determine Actual Winner and Compute Deltas
        winner_actual = self.judge.determine_winner(
            single_metrics=single_metrics,
            single_quality=single_quality,
            multi_metrics=multi_metrics,
            multi_quality=multi_quality,
        )

        delta_tokens = multi_metrics.tokens - single_metrics.tokens
        delta_cost_usd = round(multi_metrics.cost_usd - single_metrics.cost_usd, 6)
        delta_latency = round(multi_metrics.latency_seconds - single_metrics.latency_seconds, 2)
        delta_quality = round(multi_quality.overall_score - single_quality.overall_score, 2)

        print("\n" + "-" * 80)
        print(f"QUESTION RESULT: [{question.id}]")
        print(f"• Expected Winner: {question.expected_winner.value.upper()}")
        print(f"• Actual Winner:   {winner_actual.value.upper()}")
        print(f"• Quality Delta:   {delta_quality:+.2f} (Multi vs Single)")
        print(f"• Latency Delta:   {delta_latency:+.2f}s (Multi vs Single)")
        print(f"• Cost Delta:      ${delta_cost_usd:+.5f} (Multi vs Single)")
        print("-" * 80)

        return QuestionEvaluationResult(
            question_id=question.id,
            query=question.query,
            category=question.category,
            expected_winner=question.expected_winner,
            single_agent_answer=single_answer,
            single_agent_metrics=single_metrics,
            single_agent_quality=single_quality,
            multi_agent_answer=multi_answer,
            multi_agent_metrics=multi_metrics,
            multi_agent_quality=multi_quality,
            winner_actual=winner_actual,
            delta_tokens=delta_tokens,
            delta_cost_usd=delta_cost_usd,
            delta_latency_seconds=delta_latency,
            delta_quality_score=delta_quality,
            completed_at=datetime.now(UTC).isoformat(),
        )

    async def run_evaluation(
        self,
        questions: list[EvaluationQuestion] | None = None,
        run_id: str | None = None,
        resume_file: Path | str | None = None,
    ) -> EvaluationRunSummary:
        """Execute full evaluation sequence across provided questions with 30s inter-question cooldown."""
        eval_questions = questions if questions is not None else get_15_benchmark_questions()
        active_run_id = run_id or f"eval_{uuid.uuid4().hex[:10]}"
        result_file = self.output_dir / f"eval_run_{active_run_id}.json"
        results: list[QuestionEvaluationResult] = []

        if resume_file:
            res_path = Path(resume_file)
            if res_path.exists():
                try:
                    data = json.loads(res_path.read_text(encoding="utf-8"))
                    active_run_id = data.get("run_id", active_run_id)
                    result_file = res_path
                    for r_data in data.get("results", []):
                        results.append(QuestionEvaluationResult.model_validate(r_data))
                    print(
                        f"✔ Resumed existing evaluation from {result_file}: {len(results)} questions already completed."
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"[Warning] Could not resume from {resume_file}: {e}")

        print("\n" + "=" * 80)
        print(f"STARTING EVALUATION PIPELINE: Run ID '{active_run_id}'")
        print(f"• Questions Count:   {len(eval_questions)}")
        print(f"• Cooldown Break:    {self.cooldown_seconds}s between questions")
        print("• Rate Limit Target: <= 13.3 RPM (strictly below 15 RPM)")
        print(f"• Output File:       {result_file.resolve()}")
        print("=" * 80)

        completed_ids = {r.question_id for r in results}

        for idx, q in enumerate(eval_questions, 1):
            if q.id in completed_ids:
                print(f"⏭ Skipping already completed question: [{q.id}]")
                continue

            res = await self.evaluate_single_question(q)
            results.append(res)
            completed_ids.add(q.id)

            # Persist progress immediately after each question
            summary = self._build_summary(active_run_id, results, len(eval_questions))
            result_file.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
            print(
                f"✔ Progress persisted to {result_file} ({len(results)}/{len(eval_questions)} completed)"
            )

            # Mandatory 30-second cooldown break between questions (except after the last question)
            if idx < len(eval_questions):
                print(
                    f"\n⏳ [Cooldown Break] Pausing {self.cooldown_seconds:.0f} seconds to protect 15 RPM limit..."
                )
                await asyncio.sleep(self.cooldown_seconds)

        final_summary = self._build_summary(active_run_id, results, len(eval_questions))
        result_file.write_text(final_summary.model_dump_json(indent=2), encoding="utf-8")
        self._print_final_summary(final_summary)
        return final_summary

    def _build_summary(
        self,
        run_id: str,
        results: list[QuestionEvaluationResult],
        total: int,
    ) -> EvaluationRunSummary:
        """Construct aggregated metrics summary from individual question results."""
        if not results:
            return EvaluationRunSummary(
                run_id=run_id,
                created_at=datetime.now(UTC).isoformat(),
                total_questions=total,
            )

        single_wins = sum(1 for r in results if r.winner_actual == ExpectedWinner.SINGLE_AGENT)
        multi_wins = sum(1 for r in results if r.winner_actual == ExpectedWinner.MULTI_AGENT)
        ties = sum(1 for r in results if r.winner_actual == ExpectedWinner.TIE)

        avg_single_lat = round(
            sum(r.single_agent_metrics.latency_seconds for r in results) / len(results), 2
        )
        avg_multi_lat = round(
            sum(r.multi_agent_metrics.latency_seconds for r in results) / len(results), 2
        )

        avg_single_cost = round(
            sum(r.single_agent_metrics.cost_usd for r in results) / len(results), 6
        )
        avg_multi_cost = round(
            sum(r.multi_agent_metrics.cost_usd for r in results) / len(results), 6
        )

        avg_single_qual = round(
            sum(r.single_agent_quality.overall_score for r in results) / len(results), 2
        )
        avg_multi_qual = round(
            sum(r.multi_agent_quality.overall_score for r in results) / len(results), 2
        )

        return EvaluationRunSummary(
            run_id=run_id,
            created_at=datetime.now(UTC).isoformat(),
            total_questions=total,
            single_agent_wins=single_wins,
            multi_agent_wins=multi_wins,
            ties=ties,
            avg_single_latency_s=avg_single_lat,
            avg_multi_latency_s=avg_multi_lat,
            avg_single_cost_usd=avg_single_cost,
            avg_multi_cost_usd=avg_multi_cost,
            avg_single_quality=avg_single_qual,
            avg_multi_quality=avg_multi_qual,
            results=results,
        )

    def _print_final_summary(self, summary: EvaluationRunSummary) -> None:
        """Print comparative evaluation scorecard."""
        print("\n" + "=" * 80)
        print("EVALUATION PIPELINE SCORECARD SUMMARY")
        print("=" * 80)
        print(f"• Total Questions Evaluated: {len(summary.results)} / {summary.total_questions}")
        print(f"• Single Agent Wins:         {summary.single_agent_wins}")
        print(f"• Multi-Agent Wins:          {summary.multi_agent_wins}")
        print(f"• Quality / Speed Ties:      {summary.ties}")
        print("-" * 80)
        print(
            f"• Average Latency:           Single={summary.avg_single_latency_s:.2f}s | Multi={summary.avg_multi_latency_s:.2f}s"
        )
        print(
            f"• Average Cost:              Single=${summary.avg_single_cost_usd:.5f} | Multi=${summary.avg_multi_cost_usd:.5f}"
        )
        print(
            f"• Average Quality Score:     Single={summary.avg_single_quality:.2f}/5.0 | Multi={summary.avg_multi_quality:.2f}/5.0"
        )
        print("=" * 80 + "\n")


async def main_async() -> None:
    """CLI Entrypoint for running evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Evaluate Single Agent vs Multi-Agent Systems.")
    parser.add_argument(
        "--question-id", "-q", type=str, default=None, help="Run a specific question by ID"
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=None, help="Limit number of questions to evaluate"
    )
    parser.add_argument(
        "--cooldown", type=float, default=30.0, help="Break cooldown in seconds between questions"
    )
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
        default=None,
        help="Resume an existing evaluation from a JSON checkpoint file",
    )
    args = parser.parse_args()

    config = get_research_team_config()
    retrieval_service = setup_eval_retrieval_service(config)

    runner = EvaluationPipelineRunner(
        config=config,
        retrieval_service=retrieval_service,
        inter_question_cooldown_seconds=args.cooldown,
    )

    questions = get_15_benchmark_questions()
    if args.question_id:
        selected = get_benchmark_question_by_id(args.question_id)
        if not selected:
            print(f"Question '{args.question_id}' not found in benchmark dataset.")
            return
        questions = [selected]
    elif args.limit:
        questions = questions[: args.limit]

    await runner.run_evaluation(questions=questions, resume_file=args.resume)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
