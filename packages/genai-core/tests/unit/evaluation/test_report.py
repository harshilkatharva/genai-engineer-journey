from __future__ import annotations

import json
from pathlib import Path

from genai_core.evaluation import (
    EvalOutput,
    EvalReport,
    MetricResult,
    SampleEvalResult,
)


def _create_sample_result(
    sample_id: int,
    passed: bool,
    recall: float,
    accuracy: float,
) -> SampleEvalResult:
    return SampleEvalResult(
        sample_id=sample_id,
        query=f"Query {sample_id}",
        metrics={
            "recall": MetricResult(metric_name="recall", score=recall, passed=recall >= 0.8),
            "accuracy": MetricResult(
                metric_name="accuracy", score=accuracy, passed=accuracy >= 0.8
            ),
        },
        output=EvalOutput(answer=f"Answer {sample_id}"),
        passed=passed,
    )


def test_eval_report_from_results_aggregations() -> None:
    results = [
        _create_sample_result(1, passed=True, recall=1.0, accuracy=1.0),
        _create_sample_result(2, passed=True, recall=0.8, accuracy=0.9),
        _create_sample_result(3, passed=False, recall=0.0, accuracy=0.5),
    ]

    report = EvalReport.from_results(
        sample_results=results,
        duration_s=2.5,
        dataset_name="demo_ds",
        app_version="1.0.0",
    )

    assert report.total_samples == 3
    assert report.passed_samples == 2
    assert report.failed_samples == 1
    assert round(report.pass_rate, 2) == 0.67
    assert report.duration_s == 2.5

    recall_summ = report.summary["recall"]
    assert recall_summ.mean == round((1.0 + 0.8 + 0.0) / 3, 4)
    assert recall_summ.min == 0.0
    assert recall_summ.max == 1.0
    assert round(recall_summ.pass_rate, 2) == 0.67


def test_eval_report_to_json_and_save(tmp_path: Path) -> None:
    results = [_create_sample_result(1, passed=True, recall=1.0, accuracy=1.0)]
    report = EvalReport.from_results(results, dataset_name="json_test")

    json_str = report.to_json()
    parsed = json.loads(json_str)
    assert parsed["dataset_name"] == "json_test"
    assert parsed["total_samples"] == 1

    file_path = tmp_path / "report.json"
    report.save(file_path)
    assert file_path.exists()
    assert "json_test" in file_path.read_text(encoding="utf-8")


def test_eval_report_to_markdown() -> None:
    results = [
        _create_sample_result(1, passed=True, recall=1.0, accuracy=1.0),
        _create_sample_result(2, passed=False, recall=0.2, accuracy=0.1),
    ]
    report = EvalReport.from_results(results, dataset_name="md_test", app_version="2.0")

    md = report.to_markdown()
    assert "# Evaluation Report - md_test" in md
    assert "App Version:** 2.0" in md
    assert "| `recall` |" in md
    assert "Failed Samples (1)" in md
    assert "Query 2" in md


def test_eval_report_compare_regression_detection() -> None:
    # Baseline run
    base_results = [
        _create_sample_result(1, passed=True, recall=1.0, accuracy=1.0),
        _create_sample_result(2, passed=True, recall=1.0, accuracy=1.0),
    ]
    baseline_report = EvalReport.from_results(base_results, app_version="baseline")

    # Candidate run with regression on sample 2
    cand_results = [
        _create_sample_result(1, passed=True, recall=1.0, accuracy=1.0),
        _create_sample_result(2, passed=False, recall=0.5, accuracy=0.9),  # recall dropped 0.5
    ]
    candidate_report = EvalReport.from_results(cand_results, app_version="candidate")

    diff = candidate_report.compare(baseline=baseline_report, tolerance=0.1)

    assert diff.is_regression is True
    assert len(diff.regressions) == 1
    reg = diff.regressions[0]
    assert reg["sample_id"] == 2
    assert reg["metric"] == "recall"
    assert reg["baseline_score"] == 1.0
    assert reg["candidate_score"] == 0.5
    assert reg["delta"] == -0.5
    assert "recall" in diff.metric_diffs
