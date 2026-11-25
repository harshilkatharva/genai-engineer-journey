"""Telemetry tracker capturing resource consumption, costs, and worker metrics."""

from __future__ import annotations

import time
from typing import Any

from ..models import ExecutionMetrics, WorkerExecutionResult, WorkerStatus
from .pricing import PricingCalculator


class ResearchTelemetryTracker:
    """Tracks token consumption, latencies, tool calls, and monetary costs across run stages."""

    def __init__(
        self,
        pricing_calculator: PricingCalculator | None = None,
        initial_tokens: int = 0,
        initial_cost_usd: float = 0.0,
    ) -> None:
        self.pricing_calculator = pricing_calculator or PricingCalculator()
        self.initial_tokens = max(0, initial_tokens)
        self.initial_cost_usd = max(0.0, initial_cost_usd)
        self._start_time = time.perf_counter()
        self._end_time: float | None = None

        self.decomposition_metrics: dict[str, Any] = {
            "tokens": 0,
            "cost_usd": 0.0,
            "latency_ms": 0.0,
            "model": "",
        }
        self.worker_breakdown: dict[str, dict[str, Any]] = {}
        self.synthesis_metrics: dict[str, Any] = {
            "tokens": 0,
            "cost_usd": 0.0,
            "latency_ms": 0.0,
            "model": "",
        }
        self.verification_metrics: dict[str, Any] = {
            "passed": False,
            "gate1_passed": False,
            "gate2_passed": None,
            "issues_count": 0,
            "summary": "",
        }

    @property
    def elapsed_ms(self) -> float:
        """Elapsed milliseconds since tracker initialization."""
        end = self._end_time if self._end_time is not None else time.perf_counter()
        return round((end - self._start_time) * 1000.0, 2)

    def record_decomposition(
        self,
        tokens: int,
        cost_usd: float,
        latency_ms: float = 0.0,
        model: str = "",
    ) -> None:
        """Record telemetry for the Lead Agent decomposition stage."""
        tok = tokens if isinstance(tokens, int) and not isinstance(tokens, bool) else 0
        cost = (
            float(cost_usd)
            if isinstance(cost_usd, (int, float)) and not isinstance(cost_usd, bool)
            else 0.0
        )
        lat = (
            float(latency_ms)
            if isinstance(latency_ms, (int, float)) and not isinstance(latency_ms, bool)
            else 0.0
        )
        self.decomposition_metrics = {
            "tokens": max(0, tok),
            "cost_usd": max(0.0, round(cost, 6)),
            "latency_ms": max(0.0, round(lat, 2)),
            "model": str(model),
        }

    def record_worker_result(
        self,
        result: WorkerExecutionResult,
        model: str = "",
    ) -> None:
        """Record telemetry for an individual worker execution result."""
        sources_count = 0
        claims_count = 0
        if result.output:
            sources_count = len(result.output.sources)
            claims_count = len(result.output.claims)

        self.worker_breakdown[result.brief_id] = {
            "status": str(result.status),
            "tokens": result.tokens_used,
            "cost_usd": round(result.cost_usd, 6),
            "latency_ms": round(result.duration_ms, 2),
            "model": model,
            "sources_count": sources_count,
            "claims_count": claims_count,
            "error_message": result.error_message,
        }

    def record_synthesis(
        self,
        tokens: int,
        cost_usd: float,
        latency_ms: float = 0.0,
        model: str = "",
    ) -> None:
        """Record telemetry for the Lead Agent synthesis stage."""
        tok = tokens if isinstance(tokens, int) and not isinstance(tokens, bool) else 0
        cost = (
            float(cost_usd)
            if isinstance(cost_usd, (int, float)) and not isinstance(cost_usd, bool)
            else 0.0
        )
        lat = (
            float(latency_ms)
            if isinstance(latency_ms, (int, float)) and not isinstance(latency_ms, bool)
            else 0.0
        )
        self.synthesis_metrics = {
            "tokens": max(0, tok),
            "cost_usd": max(0.0, round(cost, 6)),
            "latency_ms": max(0.0, round(lat, 2)),
            "model": str(model),
        }

    def record_verification(
        self,
        passed: bool,
        gate1_passed: bool,
        gate2_passed: bool | None = None,
        issues_count: int = 0,
        summary: str = "",
        tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Record telemetry for the verification gate execution."""
        self.verification_metrics = {
            "passed": passed,
            "gate1_passed": gate1_passed,
            "gate2_passed": gate2_passed,
            "issues_count": max(0, issues_count),
            "summary": summary,
            "tokens": max(0, tokens),
            "cost_usd": max(0.0, cost_usd),
        }

    def finish(self) -> ExecutionMetrics:
        """Halt execution timer and produce aggregated ExecutionMetrics."""
        if self._end_time is None:
            self._end_time = time.perf_counter()

        duration_ms = self.elapsed_ms

        worker_tokens = sum(m.get("tokens", 0) for m in self.worker_breakdown.values())
        worker_costs = sum(m.get("cost_usd", 0.0) for m in self.worker_breakdown.values())

        verif_tokens = self.verification_metrics.get("tokens", 0)
        verif_costs = self.verification_metrics.get("cost_usd", 0.0)

        total_tokens = (
            self.initial_tokens
            + self.decomposition_metrics["tokens"]
            + worker_tokens
            + self.synthesis_metrics["tokens"]
            + verif_tokens
        )
        total_cost_usd = round(
            self.initial_cost_usd
            + self.decomposition_metrics["cost_usd"]
            + worker_costs
            + self.synthesis_metrics["cost_usd"]
            + verif_costs,
            6,
        )

        completed = 0
        partial = 0
        failed = 0
        timed_out = 0

        for m in self.worker_breakdown.values():
            st = m.get("status")
            if st == str(WorkerStatus.COMPLETED):
                completed += 1
            elif st == str(WorkerStatus.PARTIAL):
                partial += 1
            elif st == str(WorkerStatus.FAILED):
                failed += 1
            elif st == str(WorkerStatus.TIMEOUT):
                timed_out += 1

        total_workers = len(self.worker_breakdown)

        return ExecutionMetrics(
            total_tokens=total_tokens,
            total_cost_usd=total_cost_usd,
            duration_ms=duration_ms,
            worker_breakdown=self.worker_breakdown,
            total_workers=total_workers,
            completed_workers=completed,
            partial_workers=partial,
            failed_workers=failed,
            timed_out_workers=timed_out,
        )
