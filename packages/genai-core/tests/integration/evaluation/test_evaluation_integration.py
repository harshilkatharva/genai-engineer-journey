from __future__ import annotations

import pytest

from genai_core.evaluation import (
    AnswerAccuracyMetric,
    EvalDataset,
    EvalOutput,
    EvalRunner,
    EvalSample,
    EvidencePrecisionMetric,
    EvidenceRecallMetric,
    ExpectedAnswer,
    ExpectedEvidence,
    HitRateMetric,
    Metric,
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_evaluation_pipeline_workflow() -> None:
    dataset = EvalDataset(
        samples=[
            EvalSample(
                id=1,
                query="What was the name of the ship arriving at Marseilles on February 24, 1815?",
                expected_answer=ExpectedAnswer(required=["Pharaon"]),
                expected_evidence=ExpectedEvidence(
                    any_of=[["chunk_0002", "chunk_0003"]],
                ),
            ),
            EvalSample(
                id=2,
                query="How Captain Leclere died?",
                expected_answer=ExpectedAnswer(required=["brain fever"]),
                expected_evidence=ExpectedEvidence(required=["chunk_0005"]),
            ),
        ],
        metadata={"name": "montecristo_rag_eval"},
    )

    async def simulated_rag_pipeline(sample: EvalSample) -> EvalOutput:
        if "ship" in sample.query:
            return EvalOutput(
                answer="The ship that arrived was the Pharaon.",
                retrieved_chunk_ids=["chunk_0002", "chunk_9999"],
                latency_ms=120.0,
            )
        return EvalOutput(
            answer="Captain Leclere died of brain fever.",
            retrieved_chunk_ids=["chunk_0005"],
            latency_ms=95.0,
        )

    metrics: list[Metric] = [
        EvidenceRecallMetric(),
        EvidencePrecisionMetric(),
        HitRateMetric(),
        AnswerAccuracyMetric(),
    ]

    runner = EvalRunner(metrics=metrics, max_concurrency=2)
    report = await runner.run(
        dataset=dataset, target=simulated_rag_pipeline, app_version="v1.0-test"
    )

    assert report.total_samples == 2
    assert report.passed_samples == 2
    assert report.pass_rate == 1.0

    assert report.summary["evidence_recall"].mean == 1.0
    assert report.summary["answer_accuracy"].mean == 1.0
    assert report.summary["hit_rate"].mean == 1.0

    markdown = report.to_markdown()
    assert "Evaluation Report - montecristo_rag_eval" in markdown
    assert "v1.0-test" in markdown
