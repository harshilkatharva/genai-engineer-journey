from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ..models.evaluation_model import (
    EvalOutput,
    EvalReport,
    EvalSample,
    MetricResult,
    SampleEvalResult,
)
from ..observability import safe_log
from ..resilience import RateLimiter
from .dataset import EvalDataset
from .metrics.base import Metric

_logger = logging.getLogger("genai_core.evaluation")


def _read_trace_file(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    return records


class EvalRunner:
    """Evaluation runner managing concurrent execution, pacing, fault isolation, and reporting."""

    def __init__(
        self,
        metrics: list[Metric],
        max_concurrency: int = 4,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.metrics = metrics
        self.max_concurrency = max(1, max_concurrency)
        self.rate_limiter = rate_limiter

    async def _evaluate_single_sample(
        self,
        sample: EvalSample,
        target: Callable[[EvalSample], Awaitable[EvalOutput]],
        semaphore: asyncio.Semaphore,
    ) -> SampleEvalResult:
        """Run target callable on a sample and evaluate all configured metrics under concurrency control."""
        async with semaphore:
            if self.rate_limiter is not None:
                await self.rate_limiter.acquire(tokens=100)

            t0 = time.perf_counter()
            error_str: str | None = None
            try:
                output = await target(sample)
                latency_ms = (time.perf_counter() - t0) * 1000
                if output.latency_ms <= 0:
                    output.latency_ms = round(latency_ms, 2)
            except Exception as exc:  # noqa: BLE001
                latency_ms = (time.perf_counter() - t0) * 1000
                error_str = f"Target execution error: {exc}"
                safe_log(
                    _logger,
                    30,
                    f"Evaluation target failed for sample {sample.id}: {exc}",
                    exc_info=exc,
                )
                output = EvalOutput(error=error_str, latency_ms=round(latency_ms, 2))

            metric_results: dict[str, MetricResult] = {}
            for metric in self.metrics:
                try:
                    res = await metric.evaluate(sample, output)
                    metric_results[metric.name] = res
                except Exception as m_exc:  # noqa: BLE001
                    safe_log(
                        _logger, 30, f"Metric '{metric.name}' failed on sample {sample.id}: {m_exc}"
                    )
                    metric_results[metric.name] = MetricResult(
                        metric_name=metric.name,
                        score=0.0,
                        passed=False,
                        reason=f"Metric evaluation exception: {m_exc}",
                    )

            # Sample passes only if all metrics pass and no target error occurred
            overall_pass = error_str is None and all(
                m.passed is not False for m in metric_results.values()
            )

            return SampleEvalResult(
                sample_id=sample.id,
                query=sample.query,
                metrics=metric_results,
                output=output,
                passed=overall_pass,
                error=error_str,
                metadata=sample.metadata,
            )

    async def run(
        self,
        dataset: EvalDataset,
        target: Callable[[EvalSample], Awaitable[EvalOutput]],
        app_version: str | None = None,
        dataset_name: str | None = None,
    ) -> EvalReport:
        """Execute evaluation over a dataset using a live async target callable."""
        semaphore = asyncio.Semaphore(self.max_concurrency)
        start_time = time.perf_counter()

        tasks = [
            self._evaluate_single_sample(sample=sample, target=target, semaphore=semaphore)
            for sample in dataset
        ]

        sample_results = await asyncio.gather(*tasks)
        duration_s = time.perf_counter() - start_time

        return EvalReport.from_results(
            sample_results=list(sample_results),
            duration_s=duration_s,
            dataset_name=dataset_name or dataset.metadata.get("name"),
            app_version=app_version,
            metadata=dataset.metadata,
        )

    async def evaluate_traces(
        self,
        dataset: EvalDataset,
        traces: list[dict[str, Any]] | str | Path,
        trace_mapper: Callable[[dict[str, Any]], tuple[str, EvalOutput]] | None = None,
        app_version: str | None = None,
        dataset_name: str | None = None,
    ) -> EvalReport:
        """Evaluate pre-recorded offline execution traces (e.g. from JSONL tracking logs)."""
        # Load traces if path is provided
        raw_traces: list[dict[str, Any]] = []
        if isinstance(traces, (str, Path)):
            p = Path(traces)
            if not p.is_file():
                raise FileNotFoundError(f"Traces file not found: {p}")
            raw_traces = await asyncio.to_thread(_read_trace_file, p)
        else:
            raw_traces = list(traces)

        # Index traces by query or sample_id
        traces_by_query: dict[str, EvalOutput] = {}
        traces_by_id: dict[str, EvalOutput] = {}

        for record in raw_traces:
            if (
                app_version
                and record.get("app_version")
                and record.get("app_version") != app_version
            ):
                continue

            if trace_mapper is not None:
                key, out = trace_mapper(record)
                traces_by_query[key.strip().casefold()] = out
                traces_by_id[str(key).strip()] = out
            else:
                q = str(record.get("query") or "").strip().casefold()
                s_id = str(
                    record.get("id") or record.get("sample_id") or record.get("question_id") or ""
                ).strip()
                ans = record.get("llm_answer") or record.get("answer")
                raw_chunk_ids = record.get("chunk_ids") or record.get("retrieved_chunk_ids") or []
                chunk_ids = (
                    [str(c).strip() for c in raw_chunk_ids if str(c).strip()]
                    if isinstance(raw_chunk_ids, (list, tuple, set))
                    else ([str(raw_chunk_ids).strip()] if raw_chunk_ids else [])
                )
                raw_contexts = record.get("retrieved_contexts") or record.get("contexts") or []
                contexts = (
                    [str(c) for c in raw_contexts]
                    if isinstance(raw_contexts, (list, tuple))
                    else ([str(raw_contexts)] if raw_contexts else None)
                )

                out = EvalOutput(
                    answer=str(ans) if ans is not None else None,
                    retrieved_chunk_ids=chunk_ids,
                    retrieved_contexts=contexts,
                    latency_ms=float(record.get("latency_ms") or 0.0),
                    raw_response=record,
                )
                if q:
                    traces_by_query[q] = out
                if s_id:
                    traces_by_id[s_id] = out

        # Evaluate matched samples
        start_time = time.perf_counter()
        sample_results: list[SampleEvalResult] = []

        for sample in dataset:
            matched_output = traces_by_id.get(str(sample.id)) or traces_by_query.get(
                sample.query.strip().casefold()
            )
            if matched_output is None:
                # No recorded trace found for this query
                sample_results.append(
                    SampleEvalResult(
                        sample_id=sample.id,
                        query=sample.query,
                        output=EvalOutput(error="No matching trace found in log"),
                        passed=False,
                        error="Missing trace record",
                        metadata=sample.metadata,
                    )
                )
                continue

            metric_results: dict[str, MetricResult] = {}
            for metric in self.metrics:
                try:
                    res = await metric.evaluate(sample, matched_output)
                    metric_results[metric.name] = res
                except Exception as m_exc:  # noqa: BLE001
                    metric_results[metric.name] = MetricResult(
                        metric_name=metric.name,
                        score=0.0,
                        passed=False,
                        reason=f"Metric evaluation exception: {m_exc}",
                    )

            overall_pass = matched_output.error is None and all(
                m.passed is not False for m in metric_results.values()
            )
            sample_results.append(
                SampleEvalResult(
                    sample_id=sample.id,
                    query=sample.query,
                    metrics=metric_results,
                    output=matched_output,
                    passed=overall_pass,
                    metadata=sample.metadata,
                )
            )

        duration_s = time.perf_counter() - start_time
        return EvalReport.from_results(
            sample_results=sample_results,
            duration_s=duration_s,
            dataset_name=dataset_name or dataset.metadata.get("name"),
            app_version=app_version,
            metadata=dataset.metadata,
        )
