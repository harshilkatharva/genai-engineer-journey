"""Unit tests for EvaluationRateLimitController and HTTP 429 rate limit management."""

from __future__ import annotations

import time
from typing import ClassVar
from unittest.mock import AsyncMock

import pytest
from genai_core import LLMError, LLMManagerResponse

from self_improving_report_writer.core.exceptions import EvaluationRateLimitError
from self_improving_report_writer.core.rate_limit_controller import (
    EvaluationRateLimitController,
    extract_retry_delay,
    is_rate_limit_error,
)
from self_improving_report_writer.judging.prompts import (
    LLMJudgeEvaluation,
)
from self_improving_report_writer.judging.rubric_judge import RubricJudge
from self_improving_report_writer.models.report import (
    Report,
    SourceReference,
)
from self_improving_report_writer.verification.semantic_grounding_verifier import (
    SemanticGroundingVerifier,
)

# ============================================================================
# 1. Detection of HTTP 429 / RESOURCE_EXHAUSTED
# ============================================================================


def test_is_rate_limit_error_detection() -> None:
    """Verify accurate detection of diverse 429 rate-limit signatures."""
    # 1. status_code = 429
    resp_429 = LLMManagerResponse(
        error=LLMError(
            provider="google", code="rate_limit", message="Too many requests", status_code=429
        )
    )
    assert is_rate_limit_error(resp_429) is True

    # 2. code = RESOURCE_EXHAUSTED
    resp_resource = LLMManagerResponse(
        error=LLMError(provider="google", code="RESOURCE_EXHAUSTED", message="Resource exhausted")
    )
    assert is_rate_limit_error(resp_resource) is True

    # 3. Message contains Google 429 quota exhaustion string
    google_msg = "429 Resource has been exhausted (e.g. check quota). Please retry in 52.846599s."
    resp_google = LLMManagerResponse(
        error=LLMError(provider="google", code="error", message=google_msg)
    )
    assert is_rate_limit_error(resp_google) is True

    # 4. Message contains retry in/after
    resp_retry_in = LLMManagerResponse(
        error=LLMError(
            provider="google", code="error", message="Rate limit exceeded. Please retry after 15s"
        )
    )
    assert is_rate_limit_error(resp_retry_in) is True

    # 5. Headers contain Retry-After
    class MockRespWithHeaders:
        error = None
        headers: ClassVar[dict[str, str]] = {"retry-after": "30"}

    assert is_rate_limit_error(MockRespWithHeaders()) is True

    # 6. Direct Exception
    exc = Exception("429 Client Error: Too Many Requests")
    assert is_rate_limit_error(exc) is True

    # 7. Non-429 errors must return False
    resp_other = LLMManagerResponse(
        error=LLMError(
            provider="google", code="invalid_argument", message="Bad request", status_code=400
        )
    )
    assert is_rate_limit_error(resp_other) is False
    assert is_rate_limit_error(None) is False
    assert is_rate_limit_error(LLMManagerResponse(text="Success")) is False


# ============================================================================
# 2. Extraction of Recommended Retry Delays
# ============================================================================


def test_extract_retry_delay() -> None:
    """Verify parsing of floating-point and integer retry delays from error messages."""
    # Google standard floating point delay
    err1 = LLMManagerResponse(
        error=LLMError(
            provider="google",
            code="RESOURCE_EXHAUSTED",
            message="Resource has been exhausted (e.g. check quota). Please retry in 52.846599s.",
        )
    )
    assert extract_retry_delay(err1) == pytest.approx(52.846599, abs=1e-5)

    # Integer delay with 'retry after'
    err2 = LLMManagerResponse(
        error=LLMError(
            provider="google",
            code="429",
            message="Rate limit reached. Please retry after 25s.",
        )
    )
    assert extract_retry_delay(err2) == pytest.approx(25.0)

    # Full word 'seconds'
    err3 = Exception("Quota exceeded. Please retry in 40 seconds")
    assert extract_retry_delay(err3) == pytest.approx(40.0)

    # Key-value fallback
    err4 = Exception("Error payload: retry_delay: 12.5")
    assert extract_retry_delay(err4) == pytest.approx(12.5)

    # Header-based delay
    class HeaderResp:
        error = None
        headers: ClassVar[dict[str, str]] = {"retry-after": "18"}

    assert extract_retry_delay(HeaderResp()) == pytest.approx(18.0)

    # Unspecified delay uses fallback default
    err_none = LLMManagerResponse(
        error=LLMError(provider="google", code="429", message="Rate limit exceeded.")
    )
    assert extract_retry_delay(err_none, default_delay=7.5) == pytest.approx(7.5)


# ============================================================================
# 3. Automatic Retry After Delay
# ============================================================================


@pytest.mark.asyncio
async def test_automatic_retry_success() -> None:
    """Verify that controller pauses and automatically retries upon 429 response."""
    ctrl = EvaluationRateLimitController(
        max_retries=2,
        safety_buffer=0.01,
        default_backoff=0.02,
    )

    call_count = 0

    async def mock_call() -> LLMManagerResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LLMManagerResponse(
                error=LLMError(
                    provider="google",
                    code="RESOURCE_EXHAUSTED",
                    message="Please retry in 0.05s",
                )
            )
        return LLMManagerResponse(text="Success on attempt 2")

    t0 = time.monotonic()
    result = await ctrl.execute_with_retry(mock_call)
    elapsed = time.monotonic() - t0

    assert call_count == 2
    assert result.error is None
    assert result.text == "Success on attempt 2"
    # Elapsed time must account for the 0.05s + 0.01s buffer
    assert elapsed >= 0.05


# ============================================================================
# 4. Request Barrier Pauses Concurrent Requests
# ============================================================================


@pytest.mark.asyncio
async def test_concurrent_request_barrier() -> None:
    """Verify that when one request hits 429, subsequent requests pause at the barrier."""
    ctrl = EvaluationRateLimitController(
        max_retries=2,
        safety_buffer=0.01,
        default_backoff=0.05,
    )

    t_start = time.monotonic()
    task1_finish = 0.0
    task2_start = 0.0

    async def task_1() -> LLMManagerResponse:
        nonlocal task1_finish
        # Hits 429 requiring 0.10s delay
        task1_finish = time.monotonic()
        return LLMManagerResponse(
            error=LLMError(
                provider="google",
                code="RESOURCE_EXHAUSTED",
                message="Resource exhausted. Please retry in 0.10s",
            )
        )

    async def task_2() -> LLMManagerResponse:
        nonlocal task2_start
        task2_start = time.monotonic()
        return LLMManagerResponse(text="Task 2 completed")

    # Explicitly set the pause barrier (simulating task 1 encountering 429)
    await ctrl.pause_all(0.10)
    assert ctrl.is_paused is True

    # Launch task 2 while barrier is active
    res2 = await ctrl.execute_with_retry(task_2)

    assert res2.text == "Task 2 completed"
    # Task 2 must have waited out the 0.10s barrier
    assert task2_start - t_start >= 0.09


# ============================================================================
# 5. Wall-Clock Deadline Accounting & Abort
# ============================================================================


@pytest.mark.asyncio
async def test_deadline_accounting_prevents_retry_overrun() -> None:
    """Verify that if retry delay exceeds remaining wall-clock deadline, retry is aborted immediately."""
    ctrl = EvaluationRateLimitController(
        max_retries=3,
        safety_buffer=0.5,
    )

    # Deadline is 0.10 seconds in the future
    deadline = time.monotonic() + 0.10

    async def mock_call() -> LLMManagerResponse:
        # Returns delay of 5.0 seconds, which clearly breaches 0.10s deadline
        return LLMManagerResponse(
            error=LLMError(
                provider="google",
                code="RESOURCE_EXHAUSTED",
                message="Please retry in 5.0s",
            )
        )

    t0 = time.monotonic()
    resp = await ctrl.execute_with_retry(mock_call, deadline=deadline)
    elapsed = time.monotonic() - t0

    # The controller must NOT sleep for 5.0s; it must abort quickly
    assert elapsed < 0.20
    assert resp.error is not None
    assert "RESOURCE_EXHAUSTED" in resp.error.code


@pytest.mark.asyncio
async def test_deadline_expired_before_call_raises() -> None:
    """Verify that an already-expired deadline immediately raises EvaluationRateLimitError."""
    ctrl = EvaluationRateLimitController()
    expired_deadline = time.monotonic() - 1.0

    async def mock_call() -> LLMManagerResponse:
        return LLMManagerResponse(text="Should not run")

    with pytest.raises(EvaluationRateLimitError, match="Evaluation deadline exceeded"):
        await ctrl.execute_with_retry(mock_call, deadline=expired_deadline)


# ============================================================================
# 6. Handling Repeated 429 Errors
# ============================================================================


@pytest.mark.asyncio
async def test_repeated_429_retries_exhaustion() -> None:
    """Verify that consecutive 429 errors are retried up to max_retries before returning error."""
    ctrl = EvaluationRateLimitController(
        max_retries=2,
        safety_buffer=0.005,
        default_backoff=0.01,
    )

    calls = 0

    async def persistent_429() -> LLMManagerResponse:
        nonlocal calls
        calls += 1
        return LLMManagerResponse(
            error=LLMError(
                provider="google",
                code="RESOURCE_EXHAUSTED",
                message="Please retry in 0.01s",
            )
        )

    resp = await ctrl.execute_with_retry(persistent_429)

    # Initial call + 2 retries = 3 calls
    assert calls == 3
    assert resp.error is not None
    assert "RESOURCE_EXHAUSTED" in resp.error.code


# ============================================================================
# 7. Integration with SemanticGroundingVerifier & RubricJudge
# ============================================================================


@pytest.mark.asyncio
async def test_semantic_grounding_verifier_with_429_retry(
    valid_report: Report, sample_sources: list[SourceReference]
) -> None:
    """Verify SemanticGroundingVerifier transparently recovers from 429 rate limit."""
    mock_llm = AsyncMock()

    call_count = 0

    async def mock_complete_structure(*args, **kwargs) -> LLMManagerResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LLMManagerResponse(
                error=LLMError(
                    provider="google",
                    code="RESOURCE_EXHAUSTED",
                    message="Resource has been exhausted. Please retry in 0.02s",
                )
            )
        return LLMManagerResponse(
            structured_response={
                "evaluations": [
                    {
                        "claim_id": "claim_exec_1",
                        "entailment": "ENTAILS",
                        "score": 1.0,
                        "reasoning": "Direct evidence match.",
                    },
                    {
                        "claim_id": "claim_find_1",
                        "entailment": "ENTAILS",
                        "score": 1.0,
                        "reasoning": "Direct evidence match.",
                    },
                    {
                        "claim_id": "claim_find_2",
                        "entailment": "ENTAILS",
                        "score": 1.0,
                        "reasoning": "Direct evidence match.",
                    },
                    {
                        "claim_id": "claim_concl_1",
                        "entailment": "ENTAILS",
                        "score": 1.0,
                        "reasoning": "Direct evidence match.",
                    },
                ]
            }
        )

    mock_llm.complete_structure.side_effect = mock_complete_structure

    ctrl = EvaluationRateLimitController(max_retries=2, safety_buffer=0.005, default_backoff=0.01)
    verifier = SemanticGroundingVerifier(
        llm_service=mock_llm,
        rate_limit_controller=ctrl,
        check_prose_consistency=False,
    )

    sources_map = {(s.source_id, s.chunk_id): s for s in sample_sources}

    g_report = await verifier.verify_grounding(valid_report, sources_map)
    assert call_count >= 2
    assert g_report.passed is True


@pytest.mark.asyncio
async def test_rubric_judge_with_429_retry(
    valid_report: Report, sample_sources: list[SourceReference]
) -> None:
    """Verify RubricJudge transparently recovers from 429 rate limit without triggering premature fallback."""
    mock_llm = AsyncMock()

    call_count = 0

    scores = {
        "factual_accuracy": 0.95,
        "evidence_grounding": 0.95,
        "citation_correctness": 0.95,
        "completeness": 0.95,
        "relevance": 0.95,
        "clarity_organization": 0.95,
        "logical_consistency": 0.95,
    }
    raw_eval = LLMJudgeEvaluation(
        overall_feedback="High quality",
        dimension_scores=[
            {"dimension_name": k, "score": v, "passed": True, "reasoning": "Excellent"}
            for k, v in scores.items()
        ],
    )

    async def mock_complete_structure(*args, **kwargs) -> LLMManagerResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LLMManagerResponse(
                error=LLMError(
                    provider="google",
                    code="RESOURCE_EXHAUSTED",
                    message="Resource has been exhausted. Please retry in 0.02s",
                )
            )
        return LLMManagerResponse(structured_response=raw_eval)

    mock_llm.complete_structure.side_effect = mock_complete_structure

    ctrl = EvaluationRateLimitController(max_retries=2, safety_buffer=0.005, default_backoff=0.01)
    judge = RubricJudge(
        llm_service=mock_llm,
        judge_model="gemini-3.1-flash-lite",
        generator_model="gemini-3.5-flash-lite",
        rate_limit_controller=ctrl,
    )

    sources_map = {s.chunk_id: s for s in sample_sources}

    verdict = await judge.evaluate(valid_report, sources=sources_map)
    assert call_count == 2
    assert verdict.passed is True
    # Crucial: Did not mistakenly trigger fallback to generator model
    assert verdict.is_self_judged is False
