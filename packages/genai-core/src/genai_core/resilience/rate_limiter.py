from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence


def estimate_tokens(text_or_texts: str | Sequence[str]) -> int:
    """Documented heuristic for rate-limit pacing when native token counts are unavailable.

    Uses a 4-characters-per-token heuristic. This is strictly an internal rate-limiting estimate
    and is never exposed as actual usage metadata in service responses.
    """
    if isinstance(text_or_texts, str):
        return max(1, len(text_or_texts.strip()) // 4)
    total_chars = sum(len(t.strip()) for t in text_or_texts if isinstance(t, str))
    return max(1, total_chars // 4)


class AsyncTokenBucket:
    """Thread-safe and async-safe token bucket for rate-limiting.

    Note: This is an in-memory, process-local rate limiter. It coordinates coroutines
    within a single Python process and is not globally coordinated across multiple
    processes, containers, or worker nodes.
    """

    def __init__(self, capacity: float, refill_rate_per_sec: float) -> None:
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate_per_sec)
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self, now: float) -> None:
        elapsed = now - self.last_refill
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

    async def acquire(self, amount: float = 1.0) -> float:
        """Acquire tokens from the bucket, waiting asynchronously if necessary.

        Returns the number of seconds waited (0.0 if capacity was immediately available).
        """
        if amount <= 0 or self.capacity <= 0:
            return 0.0

        amount = min(amount, self.capacity)
        total_wait = 0.0

        while True:
            async with self._lock:
                now = time.monotonic()
                self._refill(now)

                if self.tokens >= amount:
                    self.tokens -= amount
                    return total_wait

                deficit = amount - self.tokens
                wait_time = deficit / self.refill_rate if self.refill_rate > 0 else 0.1

            total_wait += wait_time
            await asyncio.sleep(wait_time)


class RateLimiter:
    """Dual-dimension in-memory rate limiter enforcing RPM (Requests Per Minute)

    and TPM (Tokens Per Minute) across asynchronous operations.

    Note: This rate limiter is strictly process-local (in-memory) and protects against
    exceeding provider quotas within the current application process. It is not
    a distributed coordinator.
    """

    def __init__(
        self,
        rpm: int | None = None,
        tpm: int | None = None,
        provider_name: str = "generic",
    ) -> None:
        self.provider_name = provider_name
        self.rpm = rpm
        self.tpm = tpm

        self._req_bucket: AsyncTokenBucket | None = (
            AsyncTokenBucket(capacity=rpm, refill_rate_per_sec=rpm / 60.0)
            if rpm is not None and rpm > 0
            else None
        )
        self._token_bucket: AsyncTokenBucket | None = (
            AsyncTokenBucket(capacity=tpm, refill_rate_per_sec=tpm / 60.0)
            if tpm is not None and tpm > 0
            else None
        )

        self._pause_until: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(
        self,
        requests: int = 1,
        tokens: int = 0,
    ) -> float:
        """Acquire request and token quota, waiting non-blockingly if limits are reached.

        Returns the total deliberate delay in seconds (0.0 if capacity was available).
        """
        total_delay = 0.0

        # Handle explicit dynamic Retry-After pauses from HTTP 429 responses
        async with self._lock:
            pause_remaining = self._pause_until - time.monotonic()

        if pause_remaining > 0:
            total_delay += pause_remaining
            await asyncio.sleep(pause_remaining)

        # Acquire request budget
        if self._req_bucket is not None:
            total_delay += await self._req_bucket.acquire(float(requests))

        # Acquire token budget
        if self._token_bucket is not None and tokens > 0:
            total_delay += await self._token_bucket.acquire(float(tokens))

        return total_delay

    def record_rate_limit(self, retry_after: float | None = None) -> None:
        """Notify the limiter that an upstream provider returned HTTP 429.

        Pauses future acquisitions by the specified retry_after delay (or 5.0s default).
        """
        delay = float(retry_after) if retry_after is not None and retry_after > 0 else 5.0
        self._pause_until = max(self._pause_until, time.monotonic() + delay)
