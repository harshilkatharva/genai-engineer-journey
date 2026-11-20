"""Unit tests for ResearchTelemetryTracker."""

from __future__ import annotations

import time

from parallel_research_team.models import (
    Claim,
    SourceReference,
    WorkerExecutionResult,
    WorkerResearchOutput,
    WorkerStatus,
)
from parallel_research_team.telemetry.tracker import ResearchTelemetryTracker


def test_tracker_initialization() -> None:
    tracker = ResearchTelemetryTracker()
    assert tracker.elapsed_ms >= 0.0
    assert tracker.decomposition_metrics["tokens"] == 0
    assert tracker.decomposition_metrics["cost_usd"] == 0.0
    assert len(tracker.worker_breakdown) == 0


def test_record_decomposition() -> None:
    tracker = ResearchTelemetryTracker()
    tracker.record_decomposition(
        tokens=1200, cost_usd=0.0035, latency_ms=450.0, model="gemini-3.5-flash-lite"
    )

    assert tracker.decomposition_metrics["tokens"] == 1200
    assert tracker.decomposition_metrics["cost_usd"] == 0.0035
    assert tracker.decomposition_metrics["latency_ms"] == 450.0
    assert tracker.decomposition_metrics["model"] == "gemini-3.5-flash-lite"


def test_record_worker_results() -> None:
    tracker = ResearchTelemetryTracker()

    claim = Claim(claim_id="c1", section_id="s1", text="Claim 1")
    source = SourceReference(
        source_id="s1",
        chunk_id="chk1",
        title="Doc 1",
        content="Content 1",
    )

    res1 = WorkerExecutionResult(
        brief_id="brief_01",
        status=WorkerStatus.COMPLETED,
        output=WorkerResearchOutput(
            brief_id="brief_01",
            findings_summary="Summary 1",
            claims=[claim.model_dump()],
            sources=[source.model_dump()],
        ),
        tokens_used=1500,
        cost_usd=0.004,
        duration_ms=800.0,
    )

    res2 = WorkerExecutionResult(
        brief_id="brief_02",
        status=WorkerStatus.FAILED,
        error_message="Network error",
        tokens_used=300,
        cost_usd=0.001,
        duration_ms=200.0,
    )

    tracker.record_worker_result(res1, model="gemini-3.5-flash-lite")
    tracker.record_worker_result(res2, model="gemini-3.5-flash-lite")

    assert len(tracker.worker_breakdown) == 2
    assert tracker.worker_breakdown["brief_01"]["status"] == "completed"
    assert tracker.worker_breakdown["brief_01"]["sources_count"] == 1
    assert tracker.worker_breakdown["brief_01"]["claims_count"] == 1
    assert tracker.worker_breakdown["brief_02"]["status"] == "failed"
    assert tracker.worker_breakdown["brief_02"]["error_message"] == "Network error"


def test_record_synthesis_and_verification() -> None:
    tracker = ResearchTelemetryTracker()
    tracker.record_synthesis(
        tokens=2500, cost_usd=0.008, latency_ms=1200.0, model="gemini-3.5-flash-lite"
    )
    tracker.record_verification(
        passed=True, gate1_passed=True, gate2_passed=True, issues_count=0, summary="Clean"
    )

    assert tracker.synthesis_metrics["tokens"] == 2500
    assert tracker.verification_metrics["passed"] is True
    assert tracker.verification_metrics["gate2_passed"] is True


def test_finish_aggregation() -> None:
    tracker = ResearchTelemetryTracker()
    time.sleep(0.01)

    tracker.record_decomposition(tokens=1000, cost_usd=0.002)

    res_comp = WorkerExecutionResult(
        brief_id="brief_01",
        status=WorkerStatus.COMPLETED,
        tokens_used=2000,
        cost_usd=0.005,
    )
    res_part = WorkerExecutionResult(
        brief_id="brief_02",
        status=WorkerStatus.PARTIAL,
        tokens_used=1000,
        cost_usd=0.002,
    )
    res_fail = WorkerExecutionResult(
        brief_id="brief_03",
        status=WorkerStatus.FAILED,
        tokens_used=500,
        cost_usd=0.001,
    )
    res_to = WorkerExecutionResult(
        brief_id="brief_04",
        status=WorkerStatus.TIMEOUT,
        tokens_used=500,
        cost_usd=0.001,
    )

    tracker.record_worker_result(res_comp)
    tracker.record_worker_result(res_part)
    tracker.record_worker_result(res_fail)
    tracker.record_worker_result(res_to)

    tracker.record_synthesis(tokens=1500, cost_usd=0.004)

    metrics = tracker.finish()

    assert metrics.total_tokens == 1000 + (2000 + 1000 + 500 + 500) + 1500  # 6500
    assert metrics.total_cost_usd == round(0.002 + (0.005 + 0.002 + 0.001 + 0.001) + 0.004, 6)
    assert metrics.duration_ms > 0.0
    assert metrics.total_workers == 4
    assert metrics.completed_workers == 1
    assert metrics.partial_workers == 1
    assert metrics.failed_workers == 1
    assert metrics.timed_out_workers == 1


def test_resume_initial_tokens_and_cost() -> None:
    tracker = ResearchTelemetryTracker(initial_tokens=5000, initial_cost_usd=0.015)
    tracker.record_decomposition(tokens=500, cost_usd=0.001)
    metrics = tracker.finish()

    assert metrics.total_tokens == 5500
    assert metrics.total_cost_usd == 0.016
