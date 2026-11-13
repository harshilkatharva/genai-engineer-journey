"""Integration tests for 15-Topic Comparative Evaluation Suite and Benchmark Runner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from self_improving_report_writer.core.budget import RunBudget
from self_improving_report_writer.core.config import ReportWriterConfig
from self_improving_report_writer.eval.benchmark_runner import BenchmarkRunner
from self_improving_report_writer.eval.blinded_judge import BlindedJudge
from self_improving_report_writer.eval.dataset import (
    BenchmarkTopic,
    get_benchmark_dataset,
    get_topic_by_id,
)
from self_improving_report_writer.eval.metrics import (
    compute_comparative_summary,
)
from self_improving_report_writer.eval.runners import (
    ApproachRunResult,
    ApproachType,
)
from self_improving_report_writer.models.outcome import ReportCandidate, ReportStatus
from self_improving_report_writer.models.report import (
    Citation,
    CitationType,
    Claim,
    Report,
    ReportSection,
    SourceReference,
)
from self_improving_report_writer.models.rubric import (
    DimensionScore,
    ReportJudgeVerdict,
    SemanticGroundingReport,
)
from self_improving_report_writer.models.verification import (
    DeterministicVerificationReport,
)
from self_improving_report_writer.reflection.orchestrator import ReflectionRunResult
from self_improving_report_writer.review.acceptance_gate import AcceptanceResult

pytestmark = pytest.mark.integration


def _build_test_report(
    topic: str = "Test Topic",
    version: int = 1,
    created_by: str = "test_generator",
) -> Report:
    cite = Citation(
        citation_id="cite_1",
        source_id="src_1",
        chunk_id="chunk_1",
        citation_type=CitationType.QUOTATION,
        quoted_passage="rotated surface codes",
    )
    claim = Claim(
        claim_id="claim_1",
        section_id="sec_exec",
        text="Quantum error correction requires rotated surface codes.",
        is_factual=True,
        citations=[cite],
    )
    exec_sec = ReportSection(
        section_id="sec_exec",
        title="Executive Summary",
        content="This is the executive summary text citing [cite_1].",
        claims=[claim],
    )
    body_sec = ReportSection(
        section_id="sec_1",
        title="Introduction",
        content="This is the introduction text citing [cite_1].",
        claims=[],
    )
    concl_sec = ReportSection(
        section_id="sec_concl",
        title="Conclusion",
        content="This is the conclusion text.",
        claims=[],
    )
    return Report(
        topic=topic,
        title=f"Report on {topic}",
        executive_summary=exec_sec,
        sections=[body_sec],
        conclusion=concl_sec,
        limitations=["Preliminary simulation results."],
        version=version,
        created_by=created_by,
    )


def _build_test_verdict(
    composite_score: float = 0.90,
    passed: bool = True,
    factual_score: float = 0.90,
    evidence_score: float = 0.90,
) -> ReportJudgeVerdict:
    return ReportJudgeVerdict(
        composite_score=composite_score,
        passed=passed,
        dimension_scores={
            "factual_accuracy": DimensionScore(
                dimension_name="factual_accuracy",
                score=factual_score,
                passed=factual_score >= 0.85,
                reasoning="Factual accuracy reasoning",
            ),
            "evidence_grounding": DimensionScore(
                dimension_name="evidence_grounding",
                score=evidence_score,
                passed=evidence_score >= 0.85,
                reasoning="Evidence grounding reasoning",
            ),
            "completeness": DimensionScore(
                dimension_name="completeness",
                score=composite_score,
                passed=True,
                reasoning="Complete reasoning",
            ),
        },
        overall_feedback="Well grounded report.",
    )


def test_benchmark_dataset_integrity() -> None:
    """Verify that the 15-topic benchmark dataset satisfies all structural requirements."""
    dataset = get_benchmark_dataset()

    assert len(dataset) == 15, f"Expected 15 benchmark topics, found {len(dataset)}"

    # Check 5 domains with 3 topics each
    domains: dict[str, list[BenchmarkTopic]] = {}
    topic_ids: set[str] = set()

    for topic in dataset:
        assert topic.topic_id not in topic_ids, f"Duplicate topic ID: {topic.topic_id}"
        topic_ids.add(topic.topic_id)
        assert len(topic.title.strip()) > 5
        assert len(topic.description.strip()) > 10
        assert len(topic.key_questions) >= 2
        assert len(topic.gold_sources) >= 1

        for src in topic.gold_sources:
            assert isinstance(src, SourceReference)
            assert len(src.chunk_id) > 0
            assert len(src.content) > 20

        domains.setdefault(topic.domain, []).append(topic)

    assert len(domains) == 5, f"Expected 5 domains, found {len(domains)}: {list(domains.keys())}"
    for d, topics in domains.items():
        assert len(topics) == 3, f"Domain {d} expected 3 topics, found {len(topics)}"

    # Check get_topic_by_id
    t1 = get_topic_by_id("qc-01")
    assert t1.topic_id == "qc-01"
    assert t1.domain == "quantum_computing"

    with pytest.raises(KeyError):
        get_topic_by_id("non-existent-topic")


def test_blinded_judge_anonymization() -> None:
    """Verify that BlindedJudge strips provenance, created_by, and version."""
    judge = BlindedJudge()
    original_report = _build_test_report(
        topic="Neutral Atom Error Correction",
        version=3,
        created_by="revision_llm",
    )

    anonymized = judge.anonymize_report(original_report)

    assert anonymized.version == 1
    assert anonymized.created_by == "anonymized_candidate"
    assert anonymized.topic == original_report.topic
    assert anonymized.title == original_report.title
    assert len(anonymized.sections) == len(original_report.sections)


def test_metrics_calculation_and_research_question_resolution() -> None:
    """Verify statistical metrics aggregation and research question resolution."""
    rep_a = _build_test_report("Topic 1", 1, "gen")
    rep_b = _build_test_report("Topic 1", 2, "rev")
    rep_c = _build_test_report("Topic 1", 3, "orch")

    results = [
        ApproachRunResult(
            topic_id="qc-01",
            topic_title="Topic 1",
            approach=ApproachType.APPROACH_A,
            approach_name="Approach A",
            final_report=rep_a,
            rounds_executed=0,
            deterministic_passed=False,
            critical_defect_count=1,
            major_defect_count=2,
            composite_rubric_score=0.74,
            factual_accuracy_score=0.75,
            evidence_grounding_score=0.70,
            dimension_scores={"factual_accuracy": 0.75, "evidence_grounding": 0.70},
            tokens_consumed=1500,
            cost_usd=0.015,
            latency_seconds=3.2,
            is_accepted=False,
            terminal_status=ReportStatus.UNRESOLVED,
        ),
        ApproachRunResult(
            topic_id="qc-01",
            topic_title="Topic 1",
            approach=ApproachType.APPROACH_B,
            approach_name="Approach B",
            final_report=rep_b,
            rounds_executed=2,
            deterministic_passed=False,
            critical_defect_count=1,
            major_defect_count=1,
            composite_rubric_score=0.78,
            factual_accuracy_score=0.79,
            evidence_grounding_score=0.76,
            dimension_scores={"factual_accuracy": 0.79, "evidence_grounding": 0.76},
            tokens_consumed=3800,
            cost_usd=0.038,
            latency_seconds=7.8,
            is_accepted=False,
            terminal_status=ReportStatus.UNRESOLVED,
        ),
        ApproachRunResult(
            topic_id="qc-01",
            topic_title="Topic 1",
            approach=ApproachType.APPROACH_C,
            approach_name="Approach C",
            final_report=rep_c,
            rounds_executed=2,
            deterministic_passed=True,
            critical_defect_count=0,
            major_defect_count=0,
            composite_rubric_score=0.92,
            factual_accuracy_score=0.92,
            evidence_grounding_score=0.91,
            dimension_scores={"factual_accuracy": 0.92, "evidence_grounding": 0.91},
            tokens_consumed=4500,
            cost_usd=0.045,
            latency_seconds=9.1,
            is_accepted=True,
            terminal_status=ReportStatus.ACCEPTED,
        ),
    ]

    summary = compute_comparative_summary(results)

    assert summary.approach_a.mean_composite_score == 0.74
    assert summary.approach_a.deterministic_pass_rate == 0.0
    assert summary.approach_a.acceptance_rate == 0.0

    assert summary.approach_b.mean_composite_score == 0.78
    assert summary.approach_b.acceptance_rate == 0.0

    assert summary.approach_c.mean_composite_score == 0.92
    assert summary.approach_c.deterministic_pass_rate == 1.0
    assert summary.approach_c.acceptance_rate == 1.0

    assert summary.delta_c_vs_a_quality == 0.18
    assert summary.delta_c_vs_a_cost_usd == 0.03
    assert "YES" in summary.research_question_resolution

    md_table = summary.to_markdown_table()
    assert "Comparative Benchmark Evaluation Across 15 Topics" in md_table
    assert "Approach C (Grounded Reflection)" in md_table
    assert "Central Research Question Resolution" in md_table


@pytest.mark.asyncio
async def test_benchmark_runner_pipeline_mock(tmp_path: Path) -> None:
    """Execute the full benchmark runner pipeline across a topic subset with mock components."""
    config = ReportWriterConfig()

    mock_rep_0 = _build_test_report("Topic 1", 1, "generator")
    mock_rep_rev = _build_test_report("Topic 1", 2, "revision_llm")

    # Mock generator
    mock_gen = MagicMock()
    mock_gen.generate_report = AsyncMock(return_value=(mock_rep_0, 1200, 0.012))

    # Mock revision engine
    mock_rev = MagicMock()
    mock_rev.revise_report = AsyncMock(return_value=(mock_rep_rev, 1400, 0.014))

    # Mock deterministic verifier
    mock_det = MagicMock()
    mock_det.verify.return_value = DeterministicVerificationReport(
        passed=True,
        schema_valid=True,
        citations_valid=True,
        passages_matched=True,
        total_citations=1,
        critical_count=0,
        major_count=0,
        minor_count=0,
        issues=[],
    )

    # Mock grounding verifier
    mock_grd = MagicMock()
    mock_grd.verify_grounding = AsyncMock(
        return_value=SemanticGroundingReport(
            passed=True,
            all_claims_grounded=True,
            grounding_score=0.92,
            critical_count=0,
            major_count=0,
            minor_count=0,
            issues=[],
        )
    )

    # Mock rubric judge
    mock_verdict = _build_test_verdict(composite_score=0.91, passed=True)
    mock_judge = MagicMock()
    mock_judge.evaluate = AsyncMock(return_value=mock_verdict)

    # Mock candidate for orchestrator and acceptance gate
    mock_cand = ReportCandidate(
        round_number=1,
        report=mock_rep_rev,
        deterministic_report=mock_det.verify.return_value,
        grounding_report=mock_grd.verify_grounding.return_value,
        judge_verdict=mock_verdict,
        tokens_used=3500,
        cost_usd=0.035,
    )

    # Mock acceptance gate
    mock_gate = MagicMock()
    mock_gate.evaluate_candidate.return_value = AcceptanceResult(
        status=ReportStatus.ACCEPTED,
        is_accepted=True,
        best_candidate=mock_cand,
        rationale="All criteria met",
        unresolved_reasons=[],
    )

    budget = RunBudget(config=config)
    budget.record_usage(tokens=3500, cost_usd=0.035)
    budget.record_round_completed()
    budget_sum = budget.to_summary()

    mock_orch = MagicMock()
    mock_orch.run = AsyncMock(
        return_value=ReflectionRunResult(
            topic="Test Topic",
            best_candidate=mock_cand,
            all_candidates=[mock_cand],
            budget_summary=budget_sum,
            rounds_executed=1,
            termination_reason="ALL_ACCEPTANCE_CRITERIA_MET",
        )
    )

    runner = BenchmarkRunner(
        config=config,
        generator=mock_gen,
        revision_engine=mock_rev,
        deterministic_verifier=mock_det,
        grounding_verifier=mock_grd,
        rubric_judge=mock_judge,
        acceptance_gate=mock_gate,
        orchestrator=mock_orch,
    )

    output_dir = tmp_path / "eval_out"

    # Run on subset of 2 topics
    exec_report = await runner.run_benchmark(
        topic_ids=["qc-01", "ai-01"],
        output_dir=output_dir,
    )

    assert exec_report.num_topics_evaluated == 2
    # 2 topics * 3 approaches = 6 results
    assert len(exec_report.results) == 6

    # Verify output files
    summary_file = output_dir / "benchmark_summary.md"
    results_file = output_dir / "benchmark_results.json"

    assert summary_file.exists()
    assert results_file.exists()

    md_text = summary_file.read_text(encoding="utf-8")
    assert "Comparative Benchmark Evaluation Across 15 Topics" in md_text

    data = json.loads(results_file.read_text(encoding="utf-8"))
    assert data["num_topics_evaluated"] == 2
    assert len(data["results"]) == 6
