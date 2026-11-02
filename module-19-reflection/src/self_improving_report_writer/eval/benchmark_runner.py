"""Benchmark runner orchestrating comparative evaluation across 15 topics and 3 approaches."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from self_improving_report_writer.core.config import ReportWriterConfig, get_default_config
from self_improving_report_writer.eval.blinded_judge import BlindedJudge
from self_improving_report_writer.eval.dataset import (
    get_benchmark_dataset,
    get_topic_by_id,
)
from self_improving_report_writer.eval.metrics import (
    ComparativeEvaluationSummary,
    compute_comparative_summary,
)
from self_improving_report_writer.eval.runners import (
    ApproachRunResult,
    run_approach_a,
    run_approach_b,
    run_approach_c,
)
from self_improving_report_writer.generation.report_generator import ReportGenerator
from self_improving_report_writer.judging.rubric_judge import RubricJudge
from self_improving_report_writer.reflection.orchestrator import ReflectionLoopOrchestrator
from self_improving_report_writer.reflection.revision_engine import RevisionEngine
from self_improving_report_writer.review.acceptance_gate import AcceptanceGate
from self_improving_report_writer.verification.deterministic_verifier import (
    DeterministicVerifier,
)
from self_improving_report_writer.verification.semantic_grounding_verifier import (
    SemanticGroundingVerifier,
)

if TYPE_CHECKING:
    from genai_core import LLMService

logger = logging.getLogger(__name__)


class BenchmarkExecutionReport(BaseModel):
    """Complete output of a benchmark suite execution run."""

    num_topics_evaluated: int = Field(description="Total number of topics run")
    results: list[ApproachRunResult] = Field(description="Individual topic approach run results")
    summary: ComparativeEvaluationSummary = Field(
        description="Aggregated comparative evaluation summary"
    )


class BenchmarkRunner:
    """Coordinates the 15-topic comparative evaluation across Approaches A, B, and C."""

    def __init__(
        self,
        config: ReportWriterConfig | None = None,
        generator: ReportGenerator | None = None,
        revision_engine: RevisionEngine | None = None,
        deterministic_verifier: DeterministicVerifier | None = None,
        grounding_verifier: SemanticGroundingVerifier | None = None,
        rubric_judge: RubricJudge | None = None,
        acceptance_gate: AcceptanceGate | None = None,
        orchestrator: ReflectionLoopOrchestrator | None = None,
        llm_service: LLMService | None = None,
    ) -> None:
        self.config = config or get_default_config()
        svc = llm_service
        if svc is None and (
            generator is None
            or revision_engine is None
            or grounding_verifier is None
            or rubric_judge is None
        ):
            from genai_core import LLMService

            svc = LLMService()
        self.llm_service = svc

        self.generator = generator or ReportGenerator(
            llm_service=self.llm_service, config=self.config
        )
        self.revision_engine = revision_engine or RevisionEngine(
            llm_service=self.llm_service, config=self.config
        )
        self.deterministic_verifier = deterministic_verifier or DeterministicVerifier()
        self.grounding_verifier = grounding_verifier or SemanticGroundingVerifier(
            llm_service=self.llm_service, config=self.config
        )
        self.rubric_judge = rubric_judge or RubricJudge(
            llm_service=self.llm_service, config=self.config
        )
        self.blinded_judge = BlindedJudge(rubric_judge=self.rubric_judge)
        self.acceptance_gate = acceptance_gate or AcceptanceGate(config=self.config)

        self.orchestrator = orchestrator or ReflectionLoopOrchestrator(
            llm_service=self.llm_service,
            config=self.config,
            generator=self.generator,
            revision_engine=self.revision_engine,
            deterministic_verifier=self.deterministic_verifier,
            grounding_verifier=self.grounding_verifier,
            judge=self.rubric_judge,
        )

    async def run_benchmark(
        self,
        topic_ids: list[str] | None = None,
        output_dir: str | Path | None = None,
    ) -> BenchmarkExecutionReport:
        """Execute the benchmark suite across specified topics or the full 15-topic dataset.

        Args:
            topic_ids: Optional list of topic IDs to evaluate (defaults to all 15).
            output_dir: Optional filesystem directory to save markdown and JSON artifacts.

        Returns:
            BenchmarkExecutionReport containing raw run results and comparative summary.
        """
        all_topics = get_benchmark_dataset()
        if topic_ids:
            selected_topics = [get_topic_by_id(tid) for tid in topic_ids]
        else:
            selected_topics = all_topics

        logger.info(
            f"Starting comparative benchmark evaluation across {len(selected_topics)} topics..."
        )

        results: list[ApproachRunResult] = []

        for idx, topic in enumerate(selected_topics, 1):
            logger.info(
                f"[{idx}/{len(selected_topics)}] Evaluating topic {topic.topic_id}: '{topic.title}'..."
            )

            # 1. Approach A: No Reflection
            logger.debug(f"Running Approach A for {topic.topic_id}...")
            res_a = await run_approach_a(
                topic=topic,
                generator=self.generator,
                deterministic_verifier=self.deterministic_verifier,
                grounding_verifier=self.grounding_verifier,
                blinded_judge=self.blinded_judge,
                acceptance_gate=self.acceptance_gate,
                config=self.config,
            )
            results.append(res_a)

            # 2. Approach B: Ungrounded Self-Critique
            logger.debug(f"Running Approach B for {topic.topic_id}...")
            res_b = await run_approach_b(
                topic=topic,
                max_rounds=2,
                generator=self.generator,
                revision_engine=self.revision_engine,
                deterministic_verifier=self.deterministic_verifier,
                grounding_verifier=self.grounding_verifier,
                blinded_judge=self.blinded_judge,
                acceptance_gate=self.acceptance_gate,
                config=self.config,
            )
            results.append(res_b)

            # 3. Approach C: Grounded Reflection
            logger.debug(f"Running Approach C for {topic.topic_id}...")
            res_c = await run_approach_c(
                topic=topic,
                orchestrator=self.orchestrator,
                blinded_judge=self.blinded_judge,
                acceptance_gate=self.acceptance_gate,
                config=self.config,
            )
            results.append(res_c)

        # Compute comparative summary
        summary = compute_comparative_summary(results)
        report = BenchmarkExecutionReport(
            num_topics_evaluated=len(selected_topics),
            results=results,
            summary=summary,
        )

        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            # Save Markdown summary
            md_content = summary.to_markdown_table()
            md_file = out_path / "benchmark_summary.md"
            md_file.write_text(md_content, encoding="utf-8")

            # Save JSON report
            json_file = out_path / "benchmark_results.json"
            json_file.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            logger.info(f"Saved benchmark outputs to {out_path.resolve()}")

        return report
