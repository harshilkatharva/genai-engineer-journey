from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import patch

import pytest

from genai_core.exceptions import (
    EmbeddingDimensionError,
    EmbeddingTimeoutError,
)
from genai_core.observability import ObservabilityEvent
from genai_core.resilience import (
    AsyncTokenBucket,
    RateLimiter,
    RetryPolicy,
    estimate_tokens,
    execute_with_retry,
    extract_retry_after,
    is_retryable_error,
)
from genai_core.services import EmbeddingService

# --- Rate Limiter & Token Estimation Unit Tests ---


def test_token_estimation_heuristic() -> None:
    # 4 characters per token heuristic
    assert estimate_tokens("hello world!") == 3  # 12 chars // 4 = 3
    assert estimate_tokens(["one", "two", "three"]) == 2  # (3 + 3 + 5) // 4 = 2
    assert estimate_tokens("") == 1  # Minimum 1 token for non-empty pacing


@pytest.mark.asyncio
async def test_async_token_bucket_acquire_immediate() -> None:
    bucket = AsyncTokenBucket(capacity=10.0, refill_rate_per_sec=10.0)
    waited = await bucket.acquire(5.0)
    assert waited == 0.0
    assert bucket.tokens <= 5.1


@pytest.mark.asyncio
async def test_async_token_bucket_waits_when_exhausted() -> None:
    bucket = AsyncTokenBucket(capacity=2.0, refill_rate_per_sec=10.0)
    # Drain bucket
    await bucket.acquire(2.0)

    start = time.monotonic()
    # Acquire 1 token, refill rate is 10/sec so needs 0.1s
    await bucket.acquire(1.0)
    elapsed = time.monotonic() - start
    assert elapsed >= 0.05


@pytest.mark.asyncio
async def test_rate_limiter_rpm_and_tpm_acquire() -> None:
    limiter = RateLimiter(rpm=600, tpm=60000, provider_name="test")
    waited = await limiter.acquire(requests=1, tokens=50)
    assert waited == 0.0


@pytest.mark.asyncio
async def test_rate_limiter_retry_after_pause() -> None:
    limiter = RateLimiter(provider_name="test")
    limiter.record_rate_limit(retry_after=0.1)

    start = time.monotonic()
    waited = await limiter.acquire(requests=1)
    elapsed = time.monotonic() - start

    assert elapsed >= 0.08
    assert waited >= 0.08


# --- Error Classification Unit Tests ---


def test_is_retryable_error_selective() -> None:
    # Non-retryable errors
    assert not is_retryable_error(ValueError("Bad value"))
    assert not is_retryable_error(TypeError("Bad type"))
    assert not is_retryable_error(EmbeddingDimensionError("model", requested=100, supported=200))

    auth_err: Any = RuntimeError("unauthorized")
    auth_err.status_code = 401
    assert not is_retryable_error(auth_err)

    bad_req: Any = RuntimeError("invalid argument")
    bad_req.status_code = 400
    assert not is_retryable_error(bad_req)

    quota_err: Any = RuntimeError("insufficient quota")
    quota_err.status_code = 429
    quota_err.code = "insufficient_quota"
    assert not is_retryable_error(quota_err)

    # Retryable errors
    rate_limit_err: Any = RuntimeError("rate limit exceeded")
    rate_limit_err.status_code = 429
    assert is_retryable_error(rate_limit_err)

    server_err: Any = RuntimeError("internal error")
    server_err.status_code = 503
    assert is_retryable_error(server_err)

    timeout_err = TimeoutError("timed out")
    assert is_retryable_error(timeout_err)


def test_extract_retry_after_from_header() -> None:
    err: Any = RuntimeError("Rate limited")
    err.headers = {"Retry-After": "3.5"}
    assert extract_retry_after(err) == 3.5

    err_msg = RuntimeError("Please retry after 2.0s.")
    assert extract_retry_after(err_msg) == 2.0


# --- Retry Execution Unit Tests ---


@pytest.mark.asyncio
async def test_execute_with_retry_succeeds_first_attempt() -> None:
    policy = RetryPolicy(max_retries=2, initial_backoff=0.01)
    call_count = 0

    async def _op() -> str:
        nonlocal call_count
        call_count += 1
        return "success"

    result = await execute_with_retry(_op, policy)
    assert result == "success"
    assert call_count == 1


@pytest.mark.asyncio
async def test_execute_with_retry_recovers_after_transient_failure() -> None:
    policy = RetryPolicy(max_retries=3, initial_backoff=0.01, max_backoff=0.05, jitter=False)
    call_count = 0
    retries_recorded: list[tuple[int, float]] = []

    async def _op() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            err: Any = RuntimeError("Temporary 503 Service Unavailable")
            err.status_code = 503
            raise err
        return "recovered"

    def _on_retry(attempt: int, exc: Exception, delay: float) -> None:
        retries_recorded.append((attempt, delay))

    result = await execute_with_retry(_op, policy, on_retry=_on_retry)
    assert result == "recovered"
    assert call_count == 3
    assert len(retries_recorded) == 2


@pytest.mark.asyncio
async def test_execute_with_retry_fails_immediately_on_non_retryable() -> None:
    policy = RetryPolicy(max_retries=3, initial_backoff=0.01)
    call_count = 0

    async def _op() -> str:
        nonlocal call_count
        call_count += 1
        raise ValueError("Invalid chunk format")

    with pytest.raises(ValueError, match="Invalid chunk format"):
        await execute_with_retry(_op, policy)

    # Must NOT retry invalid requests
    assert call_count == 1


@pytest.mark.asyncio
async def test_execute_with_retry_times_out_when_deadline_exceeded() -> None:
    policy = RetryPolicy(max_retries=5, initial_backoff=0.1, timeout_seconds=0.05)

    async def _slow_op() -> str:
        await asyncio.sleep(0.2)
        return "too late"

    with pytest.raises(EmbeddingTimeoutError):
        await execute_with_retry(_slow_op, policy)


# --- EmbeddingService Resilience Integration Tests ---


class MockFailingThenSucceedingProvider:
    def __init__(self) -> None:
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return "MockFailingThenSucceeding"

    @property
    def default_model(self) -> str:
        return "test-model"

    @property
    def native_dimensions(self) -> int:
        return 384

    def validate_dimensions(self, dimensions: int) -> None:
        pass

    async def embed_batch(
        self,
        texts: list[str],
        normalize: bool = True,
    ) -> list[list[float]]:
        self.call_count += 1
        if self.call_count == 1:
            err: Any = RuntimeError("Rate limit hit 429")
            err.status_code = 429
            err.headers = {"Retry-After": "0.02"}
            raise err
        return [[0.1] * 384 for _ in texts]


@pytest.mark.asyncio
async def test_embedding_service_retries_transient_batch_failure() -> None:
    provider = MockFailingThenSucceedingProvider()
    retry_policy = RetryPolicy(max_retries=2, initial_backoff=0.01, timeout_seconds=2.0)
    service = EmbeddingService(
        provider=provider,  # type: ignore[arg-type]
        retry_policy=retry_policy,
    )

    with patch("genai_core.services.embedding_service.safe_log") as mock_safe_log:
        response = await service.embed(input=["sample document"])

        assert len(response.embeddings) == 1
        assert provider.call_count == 2  # 1 failure + 1 retry success

        # Verify EMBEDDING_RETRY event was emitted
        events = [call.kwargs.get("event") for call in mock_safe_log.call_args_list]
        assert ObservabilityEvent.EMBEDDING_RETRY in events
        assert ObservabilityEvent.EMBEDDING_REQUEST_SUCCESS in events
