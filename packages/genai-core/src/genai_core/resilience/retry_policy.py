from __future__ import annotations

import asyncio
import random
import re
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from ..exceptions.embedding_exceptions import (
    EmbeddingDimensionError,
    EmbeddingTimeoutError,
    MissingExtraDependencyError,
)

PERMANENT_ERROR_CODES = {
    "invalid_api_key",
    "authentication_error",
    "permission_denied",
    "insufficient_quota",
    "billing_not_active",
    "account_deactivated",
    "invalid_request_error",
}


@dataclass
class RetryPolicy:
    """Configurable asynchronous retry and timeout policy."""

    max_retries: int = 3
    initial_backoff: float = 0.5
    max_backoff: float = 10.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    timeout_seconds: float = 30.0


def extract_retry_after(error: Exception) -> float | None:
    """Extract Retry-After duration in seconds from exception headers or message."""
    # Check headers attribute
    headers = getattr(error, "headers", None) or getattr(
        getattr(error, "response", None), "headers", None
    )
    if headers and isinstance(headers, dict):
        retry_val = headers.get("retry-after") or headers.get("Retry-After")
        if retry_val:
            try:
                return float(retry_val)
            except (ValueError, TypeError):
                pass

    # Inspect error message for 'retry after X seconds'
    msg = str(error).lower()
    match = re.search(r"retry after\s+([0-9.]+)\s*s", msg)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    return None


def is_retryable_error(error: Exception) -> bool:
    """Determine whether an error is transient and eligible for retry."""
    # Never retry validation, dimensions, missing dependencies, or programming errors
    if isinstance(
        error,
        (
            ValueError,
            TypeError,
            EmbeddingDimensionError,
            MissingExtraDependencyError,
        ),
    ):
        return False

    # Check status code
    status_code = getattr(error, "status_code", None) or getattr(
        getattr(error, "response", None), "status_code", None
    )
    if status_code is not None:
        try:
            code = int(status_code)
            if code in {400, 401, 403, 404, 422}:
                return False
            if code in {408, 429, 500, 502, 503, 504}:
                # Permanent quota exhaustion should not be retried
                err_code = getattr(error, "code", "") or ""
                return str(err_code).lower() not in PERMANENT_ERROR_CODES
        except (ValueError, TypeError):
            pass

    # Check error message strings
    err_str = str(error).lower()
    for perm in PERMANENT_ERROR_CODES:
        if perm in err_str:
            return False

    # Check explicit retryable flag
    if getattr(error, "retryable", None) is True:
        return True

    # Timeout / connection failures
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True

    return bool(
        any(
            kw in err_str
            for kw in (
                "rate limit",
                "too many requests",
                "overloaded",
                "service unavailable",
                "gateway timeout",
                "connection reset",
                "connection refused",
                "temporarily unavailable",
                "timed out",
            )
        )
    )


async def execute_with_retry[T](
    operation: Callable[[], Coroutine[Any, Any, T]],
    policy: RetryPolicy,
    *,
    deadline: float | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> T:
    """Execute an async coroutine with bounded retries, exponential backoff, jitter, and deadline."""
    effective_deadline = (
        deadline if deadline is not None else (time.monotonic() + policy.timeout_seconds)
    )

    attempt = 0
    while True:
        now = time.monotonic()
        remaining_deadline = effective_deadline - now
        if remaining_deadline <= 0:
            raise EmbeddingTimeoutError("Embedding request deadline exceeded.")

        attempt_timeout = min(remaining_deadline, policy.timeout_seconds)

        try:
            return await asyncio.wait_for(operation(), timeout=attempt_timeout)
        except TimeoutError as err:
            if attempt >= policy.max_retries:
                raise EmbeddingTimeoutError(
                    f"Embedding request attempt timed out after {attempt_timeout:.1f}s."
                ) from err

            attempt += 1
            backoff = min(
                policy.initial_backoff * (policy.backoff_multiplier ** (attempt - 1)),
                policy.max_backoff,
            )
            if policy.jitter:
                backoff += random.uniform(0, 0.5 * backoff)

            if time.monotonic() + backoff >= effective_deadline:
                raise EmbeddingTimeoutError(
                    "Overall deadline exceeded before retry backoff completed."
                ) from err

            if on_retry:
                on_retry(attempt, err, backoff)

            await asyncio.sleep(backoff)

        except Exception as err:
            if not is_retryable_error(err) or attempt >= policy.max_retries:
                raise

            attempt += 1
            suggested_wait = extract_retry_after(err)
            if suggested_wait is not None:
                backoff = suggested_wait
            else:
                backoff = min(
                    policy.initial_backoff * (policy.backoff_multiplier ** (attempt - 1)),
                    policy.max_backoff,
                )
                if policy.jitter:
                    backoff += random.uniform(0, 0.5 * backoff)

            if time.monotonic() + backoff >= effective_deadline:
                raise EmbeddingTimeoutError(
                    "Overall deadline exceeded before retry backoff completed."
                ) from err

            if on_retry:
                on_retry(attempt, err, backoff)

            await asyncio.sleep(backoff)
