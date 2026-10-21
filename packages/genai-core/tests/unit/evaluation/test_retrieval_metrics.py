from __future__ import annotations

import pytest

from genai_core.evaluation import (
    EvalOutput,
    EvalSample,
    EvidencePrecisionMetric,
    EvidenceRecallMetric,
    ExpectedEvidence,
    HitRateMetric,
    ReciprocalRankMetric,
)


@pytest.mark.asyncio
async def test_evidence_recall_perfect_required_and_any_of() -> None:
    sample = EvalSample(
        id=1,
        query="Test query",
        expected_evidence=ExpectedEvidence(
            required=["chunk_a", "chunk_b"],
            any_of=[["chunk_c1", "chunk_c2"]],
        ),
    )
    # Output retrieved chunk_a, chunk_b, and chunk_c2 (satisfies group 1)
    output = EvalOutput(retrieved_chunk_ids=["chunk_a", "chunk_b", "chunk_c2"])

    metric = EvidenceRecallMetric()
    result = await metric.evaluate(sample, output)

    assert result.score == 1.0
    assert result.passed is True
    assert result.details["satisfied_requirements"] == 3
    assert result.details["total_requirements"] == 3


@pytest.mark.asyncio
async def test_evidence_recall_partial_and_whitespace_normalization() -> None:
    sample = EvalSample(
        id=2,
        query="Test query",
        expected_evidence=ExpectedEvidence(
            required=[" chunk_1 ", "chunk_2"],
            any_of=[["chunk_3", "chunk_4"]],
        ),
    )
    # Retrieved chunk_1 only. chunk_2 missing, any_of group unsatisfied
    output = EvalOutput(retrieved_chunk_ids=["chunk_1", "extra_chunk"])

    metric = EvidenceRecallMetric(pass_threshold=0.8)
    result = await metric.evaluate(sample, output)

    # 1 satisfied out of 3 total requirements -> 0.3333...
    assert round(result.score, 2) == 0.33
    assert result.passed is False
    assert result.details["matched_required"] == ["chunk_1"]
    assert result.details["missing_required"] == ["chunk_2"]


@pytest.mark.asyncio
async def test_evidence_precision_metric() -> None:
    sample = EvalSample(
        id=3,
        query="Test query",
        expected_evidence=ExpectedEvidence(required=["c1", "c2"]),
    )
    output = EvalOutput(retrieved_chunk_ids=["c1", "c2", "c3", "c4"])

    metric = EvidencePrecisionMetric(pass_threshold=0.5)
    result = await metric.evaluate(sample, output)

    # 2 out of 4 retrieved chunks are expected
    assert result.score == 0.5
    assert result.passed is True


@pytest.mark.asyncio
async def test_hit_rate_metric() -> None:
    sample = EvalSample(
        id=4,
        query="Test query",
        expected_evidence=ExpectedEvidence(required=["c1", "c2"]),
    )
    metric = HitRateMetric()

    # Hit case
    hit_out = EvalOutput(retrieved_chunk_ids=["other", "c2"])
    res1 = await metric.evaluate(sample, hit_out)
    assert res1.score == 1.0
    assert res1.passed is True

    # Miss case
    miss_out = EvalOutput(retrieved_chunk_ids=["other1", "other2"])
    res2 = await metric.evaluate(sample, miss_out)
    assert res2.score == 0.0
    assert res2.passed is False


@pytest.mark.asyncio
async def test_reciprocal_rank_metric() -> None:
    sample = EvalSample(
        id=5,
        query="Test query",
        expected_evidence=ExpectedEvidence(required=["c_target"]),
    )
    metric = ReciprocalRankMetric()

    # First rank: 1 / 1 = 1.0
    out_rank1 = EvalOutput(retrieved_chunk_ids=["c_target", "c2"])
    res1 = await metric.evaluate(sample, out_rank1)
    assert res1.score == 1.0

    # Third rank: 1 / 3 = 0.333...
    out_rank3 = EvalOutput(retrieved_chunk_ids=["c1", "c2", "c_target"])
    res2 = await metric.evaluate(sample, out_rank3)
    assert round(res2.score, 3) == 0.333
