from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .metric_result import EvalDiff, MetricSummary, SampleEvalResult


def _percentile(values: list[float], p: float) -> float:
    """Calculate percentile p (0..1) on a list of floats."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * p
    f = int(k)
    c = f + 1
    if c < len(sorted_v):
        return sorted_v[f] + (k - f) * (sorted_v[c] - sorted_v[f])
    return sorted_v[f]


class EvalReport(BaseModel):
    """Complete evaluation report with statistical aggregations, diagnostic samples, and export tools."""

    dataset_name: str | None = None
    app_version: str | None = None
    summary: dict[str, MetricSummary] = Field(default_factory=dict)
    sample_results: list[SampleEvalResult] = Field(default_factory=list)
    total_samples: int = 0
    passed_samples: int = 0
    failed_samples: int = 0
    pass_rate: float = 0.0
    duration_s: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_results(
        cls,
        sample_results: list[SampleEvalResult],
        duration_s: float = 0.0,
        dataset_name: str | None = None,
        app_version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvalReport:
        """Construct an EvalReport with calculated statistical aggregations."""
        total = len(sample_results)
        passed_count = sum(1 for r in sample_results if r.passed)
        failed_count = total - passed_count
        pass_rate = (passed_count / total) if total > 0 else 0.0

        # Collect scores per metric
        metric_scores: dict[str, list[float]] = {}
        metric_passes: dict[str, int] = {}
        for r in sample_results:
            for name, m_res in r.metrics.items():
                if name not in metric_scores:
                    metric_scores[name] = []
                    metric_passes[name] = 0
                metric_scores[name].append(m_res.score)
                if m_res.passed is True:
                    metric_passes[name] += 1

        summaries: dict[str, MetricSummary] = {}
        for name, scores in metric_scores.items():
            count = len(scores)
            mean_val = (sum(scores) / count) if count > 0 else 0.0
            p50_val = _percentile(scores, 0.50)
            p95_val = _percentile(scores, 0.95)
            min_val = min(scores) if scores else 0.0
            max_val = max(scores) if scores else 0.0
            pass_rt = (metric_passes[name] / count) if count > 0 else 0.0

            summaries[name] = MetricSummary(
                metric_name=name,
                mean=round(mean_val, 4),
                p50=round(p50_val, 4),
                p95=round(p95_val, 4),
                min=round(min_val, 4),
                max=round(max_val, 4),
                pass_rate=round(pass_rt, 4),
                total=count,
            )

        return cls(
            dataset_name=dataset_name,
            app_version=app_version,
            summary=summaries,
            sample_results=sample_results,
            total_samples=total,
            passed_samples=passed_count,
            failed_samples=failed_count,
            pass_rate=round(pass_rate, 4),
            duration_s=round(duration_s, 3),
            metadata=metadata or {},
        )

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        """Serialize report to a JSON string or file."""
        dumped = json.dumps(self.model_dump(), indent=indent, ensure_ascii=False)
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(dumped, encoding="utf-8")
        return dumped

    def save(self, path: str | Path) -> None:
        """Alias for to_json saving to file."""
        self.to_json(path=path)

    def to_markdown(self) -> str:
        """Generate a formatted markdown report suitable for CI/CD comments and terminal logs."""
        lines: list[str] = [
            f"# Evaluation Report{' - ' + self.dataset_name if self.dataset_name else ''}",
            "",
            f"- **App Version:** {self.app_version or 'N/A'}",
            f"- **Total Samples:** {self.total_samples}",
            f"- **Overall Pass Rate:** {self.pass_rate * 100:.1f}% ({self.passed_samples}/{self.total_samples})",
            f"- **Duration:** {self.duration_s:.2f}s",
            "",
            "## Metric Summary",
            "",
            "| Metric | Mean | p50 | p95 | Min | Max | Pass Rate |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for name, summ in sorted(self.summary.items()):
            lines.append(
                f"| `{name}` | {summ.mean:.3f} | {summ.p50:.3f} | {summ.p95:.3f} | {summ.min:.3f} | {summ.max:.3f} | {summ.pass_rate * 100:.1f}% |"
            )

        # Highlight failing samples if any exist
        failures = [r for r in self.sample_results if not r.passed]
        if failures:
            lines.extend(
                [
                    "",
                    f"## Failed Samples ({len(failures)})",
                    "",
                    "| Sample ID | Query | Failing Metric | Reason / Error |",
                    "| :--- | :--- | :--- | :--- |",
                ]
            )
            for f in failures[:15]:  # limit to top 15 in markdown
                failing_metrics = [
                    f"`{k}` ({v.score:.2f})"
                    for k, v in f.metrics.items()
                    if v.passed is False or v.score < 1.0
                ]
                reason = (
                    f.error
                    or "; ".join(
                        v.reason for v in f.metrics.values() if v.reason and v.passed is False
                    )
                    or "Failed threshold"
                )
                safe_query = f.query[:60] + "..." if len(f.query) > 60 else f.query
                lines.append(
                    f"| {f.sample_id} | {safe_query} | {', '.join(failing_metrics) or 'N/A'} | {reason} |"
                )

        return "\n".join(lines)

    def compare(self, baseline: EvalReport, tolerance: float = 0.05) -> EvalDiff:
        """Compare current report against a baseline report to detect regressions."""
        metric_diffs: dict[str, float] = {}
        for name, current_summ in self.summary.items():
            if name in baseline.summary:
                diff = current_summ.mean - baseline.summary[name].mean
                metric_diffs[name] = round(diff, 4)

        # Map baseline samples by sample_id
        baseline_by_id = {s.sample_id: s for s in baseline.sample_results}
        regressions: list[dict[str, Any]] = []

        for candidate_s in self.sample_results:
            base_s = baseline_by_id.get(candidate_s.sample_id)
            if not base_s:
                continue

            for m_name, cand_m in candidate_s.metrics.items():
                base_m = base_s.metrics.get(m_name)
                if not base_m:
                    continue

                if cand_m.score < (base_m.score - tolerance):
                    regressions.append(
                        {
                            "sample_id": candidate_s.sample_id,
                            "query": candidate_s.query,
                            "metric": m_name,
                            "baseline_score": base_m.score,
                            "candidate_score": cand_m.score,
                            "delta": round(cand_m.score - base_m.score, 4),
                        }
                    )

        return EvalDiff(
            metric_diffs=metric_diffs,
            regressions=regressions,
            is_regression=len(regressions) > 0,
        )
