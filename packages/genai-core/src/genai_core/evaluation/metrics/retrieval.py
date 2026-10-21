from __future__ import annotations

from typing import Any

from ...models.evaluation_model import EvalOutput, EvalSample, MetricResult


def _normalize_id(chunk_id: Any) -> str:
    """Normalize chunk identifier by stripping whitespace and lowercasing."""
    if not isinstance(chunk_id, str):
        return str(chunk_id).strip()
    return chunk_id.strip()


class EvidenceRecallMetric:
    """Evaluates evidence recall against required and alternative (any_of) chunk sets."""

    def __init__(self, name: str = "evidence_recall", pass_threshold: float = 1.0) -> None:
        self.name = name
        self.pass_threshold = pass_threshold

    async def evaluate(self, sample: EvalSample, output: EvalOutput) -> MetricResult:
        if sample.expected_evidence is None:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                reason="No expected evidence specified for this sample.",
            )

        retrieved = {
            _normalize_id(c) for c in (output.retrieved_chunk_ids or []) if _normalize_id(c)
        }
        required = {_normalize_id(c) for c in sample.expected_evidence.required if _normalize_id(c)}
        any_of_groups = [
            {_normalize_id(c) for c in group if _normalize_id(c)}
            for group in sample.expected_evidence.any_of
            if any(_normalize_id(c) for c in group)
        ]

        total_reqs = len(required) + len(any_of_groups)
        if total_reqs == 0:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                reason="Empty evidence requirements.",
            )

        matched_required = required & retrieved
        missing_required = required - retrieved

        satisfied_any_of_count = 0
        any_of_details: list[dict[str, Any]] = []
        for idx, group in enumerate(any_of_groups, start=1):
            matched = group & retrieved
            satisfied = bool(matched)
            if satisfied:
                satisfied_any_of_count += 1
            any_of_details.append(
                {
                    "group": idx,
                    "expected": sorted(group),
                    "matched": sorted(matched),
                    "satisfied": satisfied,
                }
            )

        satisfied_total = len(matched_required) + satisfied_any_of_count
        recall = satisfied_total / total_reqs

        passed = recall >= self.pass_threshold
        reason = (
            f"Recall {recall:.2f} ({satisfied_total}/{total_reqs} evidence requirements satisfied)"
        )

        return MetricResult(
            metric_name=self.name,
            score=recall,
            passed=passed,
            reason=reason,
            details={
                "matched_required": sorted(matched_required),
                "missing_required": sorted(missing_required),
                "any_of_results": any_of_details,
                "retrieved_count": len(retrieved),
                "total_requirements": total_reqs,
                "satisfied_requirements": satisfied_total,
            },
        )


class EvidencePrecisionMetric:
    """Evaluates precision: fraction of retrieved chunks that are relevant."""

    def __init__(self, name: str = "evidence_precision", pass_threshold: float = 0.5) -> None:
        self.name = name
        self.pass_threshold = pass_threshold

    async def evaluate(self, sample: EvalSample, output: EvalOutput) -> MetricResult:
        if sample.expected_evidence is None:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                reason="No expected evidence specified.",
            )

        retrieved = [
            _normalize_id(c) for c in (output.retrieved_chunk_ids or []) if _normalize_id(c)
        ]
        if not retrieved:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                reason="No chunks retrieved.",
            )

        all_expected = {_normalize_id(c) for c in sample.expected_evidence.required}
        for grp in sample.expected_evidence.any_of:
            all_expected.update({_normalize_id(c) for c in grp})

        matched = [c for c in retrieved if c in all_expected]
        precision = len(matched) / len(retrieved)

        return MetricResult(
            metric_name=self.name,
            score=precision,
            passed=precision >= self.pass_threshold,
            reason=f"Precision {precision:.2f} ({len(matched)}/{len(retrieved)} retrieved chunks relevant)",
            details={
                "retrieved": retrieved,
                "matched": matched,
                "precision": precision,
            },
        )


class HitRateMetric:
    """Evaluates whether at least one relevant chunk was retrieved."""

    def __init__(self, name: str = "hit_rate") -> None:
        self.name = name

    async def evaluate(self, sample: EvalSample, output: EvalOutput) -> MetricResult:
        if sample.expected_evidence is None:
            return MetricResult(metric_name=self.name, score=1.0, passed=True)

        retrieved = {
            _normalize_id(c) for c in (output.retrieved_chunk_ids or []) if _normalize_id(c)
        }
        all_expected = {_normalize_id(c) for c in sample.expected_evidence.required}
        for grp in sample.expected_evidence.any_of:
            all_expected.update({_normalize_id(c) for c in grp})

        hit = bool(retrieved & all_expected) if all_expected else True
        score = 1.0 if hit else 0.0

        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=hit,
            reason="Hit found" if hit else "No relevant chunks found in retrieved set",
        )


class ReciprocalRankMetric:
    """Evaluates Mean Reciprocal Rank (MRR) for ranked retrieval chunks."""

    def __init__(self, name: str = "reciprocal_rank") -> None:
        self.name = name

    async def evaluate(self, sample: EvalSample, output: EvalOutput) -> MetricResult:
        if sample.expected_evidence is None:
            return MetricResult(metric_name=self.name, score=1.0, passed=True)

        retrieved = [
            _normalize_id(c) for c in (output.retrieved_chunk_ids or []) if _normalize_id(c)
        ]
        all_expected = {_normalize_id(c) for c in sample.expected_evidence.required}
        for grp in sample.expected_evidence.any_of:
            all_expected.update({_normalize_id(c) for c in grp})

        for rank, chunk_id in enumerate(retrieved, start=1):
            if chunk_id in all_expected:
                rr = 1.0 / rank
                return MetricResult(
                    metric_name=self.name,
                    score=rr,
                    passed=True,
                    reason=f"First relevant chunk found at rank {rank}",
                    details={"first_rank": rank, "chunk_id": chunk_id},
                )

        return MetricResult(
            metric_name=self.name,
            score=0.0,
            passed=False,
            reason="No relevant chunks found in retrieved list",
            details={"first_rank": None},
        )
