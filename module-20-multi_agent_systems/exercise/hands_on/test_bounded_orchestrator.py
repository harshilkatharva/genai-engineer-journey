import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, Literal

import pytest
from pydantic import BaseModel, Field, model_validator

# --- Strictly Typed Contracts with Validation ---


class SubTask(BaseModel):
    """Input contract for delegating work with bounded steps."""

    id: str = Field(..., min_length=1, description="Unique task identifier")
    objective: str = Field(..., min_length=3, description="Precise objective")
    output_format: str = Field(..., description="Expected output schema")
    boundaries: str = Field(default="", description="Constraints or forbidden sources")
    max_steps: int = Field(default=6, ge=1, le=15, description="Bounded step limit [1, 15]")


class WorkerResult(BaseModel):
    """Output contract returned by workers to orchestrator."""

    task_id: str = Field(..., min_length=1)
    status: Literal["ok", "failed", "timeout"] = Field(..., description="Execution status")
    summary: str = Field(default="", description="Findings summary or error details")
    sources: list[str] = Field(default_factory=list, description="Citations")
    steps_used: int = Field(default=0, ge=0, description="Steps consumed")
    tokens_used: int = Field(default=0, ge=0, description="Tokens charged")

    @model_validator(mode="after")
    def validate_summary_for_ok_status(self) -> "WorkerResult":
        if self.status == "ok" and not self.summary.strip():
            raise ValueError(
                "Successful worker result (status='ok') must have a non-empty summary."
            )
        return self


# Dependency injection: Output validator
def validate_worker_result(raw: Any, expected_task_id: str) -> WorkerResult:
    """Validates raw output against WorkerResult contract and verifies task ID alignment."""
    if isinstance(raw, dict):
        result = WorkerResult.model_validate(raw)
    elif isinstance(raw, WorkerResult):
        result = raw
    else:
        raise TypeError(f"Worker returned invalid type: {type(raw).__name__}")

    if result.task_id != expected_task_id:
        raise ValueError(f"Task ID mismatch: expected '{expected_task_id}', got '{result.task_id}'")
    return result


@dataclass
class Budget:
    max_tokens: int
    used: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def charge(self, n: int) -> bool:
        async with self.lock:
            if self.used + n > self.max_tokens:
                return False
            self.used += n
            return True


async def run_worker(
    task: SubTask,
    worker_fn: Callable[[SubTask], Coroutine[Any, Any, Any]],
    budget: Budget,
    timeout_s: float,
    validator: Callable[[Any, str], WorkerResult] = validate_worker_result,
) -> WorkerResult:
    """Bounded worker wrapper with output validation: rejects malformed outputs, never raises."""
    try:
        raw_output = await asyncio.wait_for(worker_fn(task), timeout=timeout_s)
        result = validator(raw_output, task.id)
    except TimeoutError:
        return WorkerResult(task_id=task.id, status="timeout")
    except Exception as exc:  # noqa: BLE001  # isolate worker exceptions & validation errors
        return WorkerResult(task_id=task.id, status="failed", summary=f"Rejected: {str(exc)[:200]}")

    if not await budget.charge(result.tokens_used):
        return WorkerResult(task_id=task.id, status="failed", summary="budget exceeded")
    return result


async def orchestrate(
    subtasks: list[SubTask],
    worker_fn: Callable[[SubTask], Coroutine[Any, Any, Any]],
    budget: Budget,
    max_parallel: int = 3,
    timeout_s: float = 60.0,
    validator: Callable[[Any, str], WorkerResult] = validate_worker_result,
) -> list[WorkerResult]:
    sem = asyncio.Semaphore(max_parallel)  # concurrency cap

    async def guarded(t: SubTask) -> WorkerResult:
        async with sem:
            return await run_worker(t, worker_fn, budget, timeout_s, validator)

    return list(await asyncio.gather(*(guarded(t) for t in subtasks)))


# --- Pytest Test Suite ---


@pytest.mark.asyncio
async def test_failure_isolation():
    """Verify worker exception is converted to failed status without crashing orchestrator."""

    async def worker_with_crash(task: SubTask) -> WorkerResult:
        if task.id == "fail_task":
            raise RuntimeError("Unexpected worker crash")
        return WorkerResult(task_id=task.id, status="ok", summary="completed successfully")

    tasks = [
        SubTask(id="fail_task", objective="crash", output_format="text"),
        SubTask(id="ok_task", objective="work", output_format="text"),
    ]
    results = {r.task_id: r for r in await orchestrate(tasks, worker_with_crash, Budget(1000))}
    assert results["fail_task"].status == "failed"
    assert "Unexpected worker crash" in results["fail_task"].summary
    assert results["ok_task"].status == "ok"


@pytest.mark.asyncio
async def test_timeout():
    """Verify worker exceeding timeout_s is cancelled and assigned timeout status."""

    async def slow_worker(task: SubTask) -> WorkerResult:
        await asyncio.sleep(0.2)
        return WorkerResult(task_id=task.id, status="ok", summary="slow result")

    tasks = [SubTask(id="slow_task", objective="hang", output_format="text")]
    results = await orchestrate(tasks, slow_worker, Budget(1000), timeout_s=0.05)
    assert results[0].status == "timeout"


@pytest.mark.asyncio
async def test_concurrency_cap():
    """Verify active concurrent workers never exceed max_parallel."""
    active, max_active = 0, 0

    async def tracked_worker(task: SubTask) -> WorkerResult:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return WorkerResult(task_id=task.id, status="ok", summary="tracked")

    tasks = [SubTask(id=str(i), objective="run", output_format="text") for i in range(6)]
    await orchestrate(tasks, tracked_worker, Budget(1000), max_parallel=2)
    assert max_active == 2


@pytest.mark.asyncio
async def test_budget_enforcement():
    """Verify token usage exceeding budget marks the worker as failed."""

    async def costly_worker(task: SubTask) -> WorkerResult:
        return WorkerResult(task_id=task.id, status="ok", summary="costly", tokens_used=60)

    tasks = [
        SubTask(id="task_1", objective="run", output_format="text"),
        SubTask(id="task_2", objective="run", output_format="text"),
    ]
    results = await orchestrate(tasks, costly_worker, Budget(max_tokens=100), max_parallel=1)
    assert results[0].status == "ok"
    assert results[1].status == "failed"
    assert results[1].summary == "budget exceeded"


@pytest.mark.asyncio
async def test_output_validation_rejects_malformed():
    """Task 3: Verify malformed worker outputs (invalid status, negative tokens, empty summary) are rejected."""
    budget = Budget(1000)
    task = SubTask(id="task_val", objective="Validate output", output_format="text")

    # 1. Invalid status value
    async def worker_bad_status(t: SubTask):
        return {"task_id": t.id, "status": "completed", "summary": "Done"}

    res = await run_worker(task, worker_bad_status, budget, timeout_s=5.0)
    assert res.status == "failed"
    assert "Rejected" in res.summary

    # 2. Empty summary with status="ok"
    async def worker_empty_summary(t: SubTask):
        return {"task_id": t.id, "status": "ok", "summary": "  "}

    res2 = await run_worker(task, worker_empty_summary, budget, timeout_s=5.0)
    assert res2.status == "failed"
    assert "non-empty summary" in res2.summary

    # 3. Negative token count
    async def worker_negative_tokens(t: SubTask):
        return {"task_id": t.id, "status": "ok", "summary": "Valid", "tokens_used": -15}

    res3 = await run_worker(task, worker_negative_tokens, budget, timeout_s=5.0)
    assert res3.status == "failed"
    assert "greater than or equal to 0" in res3.summary


@pytest.mark.asyncio
async def test_output_validation_accepts_valid():
    """Task 3: Verify conforming dictionary output is validated into WorkerResult."""
    budget = Budget(1000)
    task = SubTask(id="task_val2", objective="Validate conforming output", output_format="text")

    async def worker_conforming(t: SubTask):
        return {"task_id": t.id, "status": "ok", "summary": "All good", "tokens_used": 25}

    res = await run_worker(task, worker_conforming, budget, timeout_s=5.0)
    assert res.status == "ok"
    assert res.summary == "All good"
    assert res.tokens_used == 25
