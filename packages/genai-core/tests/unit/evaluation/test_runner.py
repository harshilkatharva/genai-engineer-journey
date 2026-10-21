from __future__ import annotations

from pathlib import Path

import pytest

from genai_core.evaluation import (
    AnswerAccuracyMetric,
    EvalDataset,
    EvalOutput,
    EvalRunner,
    EvalSample,
    EvidenceRecallMetric,
    ExpectedAnswer,
    ExpectedEvidence,
    Metric,
)
from genai_core.resilience import RateLimiter


@pytest.mark.asyncio
async def test_eval_runner_live_target() -> None:
    dataset = EvalDataset(
        samples=[
            EvalSample(
                id=1,
                query="What is the capital of Japan?",
                expected_answer=ExpectedAnswer(required=["Tokyo"]),
                expected_evidence=ExpectedEvidence(required=["chunk_tokyo"]),
            ),
            EvalSample(
                id=2,
                query="What is the capital of Italy?",
                expected_answer=ExpectedAnswer(required=["Rome"]),
                expected_evidence=ExpectedEvidence(required=["chunk_rome"]),
            ),
        ]
    )

    async def mock_target(sample: EvalSample) -> EvalOutput:
        if "Japan" in sample.query:
            return EvalOutput(
                answer="The capital of Japan is Tokyo.",
                retrieved_chunk_ids=["chunk_tokyo"],
            )
        return EvalOutput(
            answer="The capital of Italy is Rome.",
            retrieved_chunk_ids=["chunk_rome"],
        )

    metrics: list[Metric] = [
        AnswerAccuracyMetric(),
        EvidenceRecallMetric(),
    ]
    runner = EvalRunner(metrics=metrics, max_concurrency=2)
    report = await runner.run(dataset=dataset, target=mock_target, app_version="1.0.0")

    assert report.total_samples == 2
    assert report.passed_samples == 2
    assert report.pass_rate == 1.0
    assert report.summary["answer_accuracy"].mean == 1.0
    assert report.summary["evidence_recall"].mean == 1.0
    assert report.sample_results[0].output.latency_ms > 0


@pytest.mark.asyncio
async def test_eval_runner_target_error_isolation() -> None:
    dataset = EvalDataset(
        samples=[
            EvalSample(id=1, query="Good query", expected_answer=ExpectedAnswer(required=["OK"])),
            EvalSample(
                id=2, query="Failing query", expected_answer=ExpectedAnswer(required=["OK"])
            ),
        ]
    )

    async def faulty_target(sample: EvalSample) -> EvalOutput:
        if sample.id == 2:
            raise RuntimeError("Database connection timed out")
        return EvalOutput(answer="OK")

    runner = EvalRunner(metrics=[AnswerAccuracyMetric()])
    report = await runner.run(dataset=dataset, target=faulty_target)

    assert report.total_samples == 2
    assert report.passed_samples == 1
    assert report.failed_samples == 1

    failed_sample = next(s for s in report.sample_results if s.sample_id == 2)
    assert failed_sample.passed is False
    assert "Database connection timed out" in (failed_sample.error or "")


@pytest.mark.asyncio
async def test_eval_runner_evaluate_traces_jsonl(tmp_path: Path) -> None:
    # Mimics module-09 query_performance_tracker.jsonl output
    trace_file = tmp_path / "query_performance_tracker.jsonl"
    trace_file.write_text(
        '{"app_version": "v1", "query": "Who is Edmond?", "chunk_ids": ["c10"], "llm_answer": "He is the Count of Monte Cristo"}\n'
        '{"app_version": "v1", "query": "What ship?", "chunk_ids": ["c01"], "llm_answer": "The ship is Pharaon"}\n',
        encoding="utf-8",
    )

    dataset = EvalDataset(
        samples=[
            EvalSample(
                id=1,
                query="Who is Edmond?",
                expected_answer=ExpectedAnswer(required=["Monte Cristo"]),
                expected_evidence=ExpectedEvidence(required=["c10"]),
            ),
            EvalSample(
                id=2,
                query="What ship?",
                expected_answer=ExpectedAnswer(required=["Pharaon"]),
                expected_evidence=ExpectedEvidence(required=["c01"]),
            ),
        ]
    )

    runner = EvalRunner(
        metrics=[AnswerAccuracyMetric(), EvidenceRecallMetric()],
    )
    report = await runner.evaluate_traces(
        dataset=dataset,
        traces=trace_file,
        app_version="v1",
    )

    assert report.total_samples == 2
    assert report.passed_samples == 2
    assert report.summary["answer_accuracy"].mean == 1.0
    assert report.summary["evidence_recall"].mean == 1.0


@pytest.mark.asyncio
async def test_eval_runner_with_rate_limiter() -> None:
    rate_limiter = RateLimiter(rpm=600, tpm=100000)
    dataset = EvalDataset(
        samples=[
            EvalSample(id=1, query="Pacing test", expected_answer=ExpectedAnswer(required=["Yes"]))
        ]
    )

    async def quick_target(sample: EvalSample) -> EvalOutput:
        return EvalOutput(answer="Yes")

    runner = EvalRunner(
        metrics=[AnswerAccuracyMetric()],
        rate_limiter=rate_limiter,
    )
    report = await runner.run(dataset=dataset, target=quick_target)
    assert report.passed_samples == 1
