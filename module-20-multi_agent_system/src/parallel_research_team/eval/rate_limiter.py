"""Asynchronous rate limiter guaranteeing strict compliance with LLM rate limits (<= 15 RPM)."""

from __future__ import annotations

import asyncio
import time
from typing import Self

from genai_core.observability import get_logger

_logger = get_logger("parallel_research_team.eval.rate_limiter")


class AsyncRateLimiter:
    """Thread-safe and asyncio-safe rate limiter implementing sliding-window throttling.

    Guarantees:
    1. Maximum requests per 60-second sliding window does not exceed max_rpm.
    2. Minimum delay between consecutive requests is at least min_interval_seconds.
    """

    def __init__(
        self,
        max_rpm: int = 12,
        min_interval_seconds: float = 4.5,
    ) -> None:
        self.max_rpm = max(1, max_rpm)
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self._timestamps: list[float] = []
        self._last_request_time: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """Acquire permission to make an API call, sleeping if necessary to enforce rate limits.

        Returns:
            The number of seconds waited (0.0 if no sleep was required).
        """
        async with self._lock:
            total_waited = 0.0

            while True:
                now = time.monotonic()

                # 1. Purge timestamps older than 60 seconds
                cutoff = now - 60.0
                self._timestamps = [t for t in self._timestamps if t > cutoff]

                # 2. Check RPM ceiling
                rpm_wait = 0.0
                if len(self._timestamps) >= self.max_rpm:
                    oldest = self._timestamps[0]
                    rpm_wait = max(0.0, 60.0 - (now - oldest) + 0.1)

                # 3. Check min interval spacing
                elapsed_since_last = now - self._last_request_time
                interval_wait = max(0.0, self.min_interval_seconds - elapsed_since_last)

                wait_time = max(rpm_wait, interval_wait)
                if wait_time <= 0.01:
                    # Proceed with call
                    current_time = time.monotonic()
                    self._timestamps.append(current_time)
                    self._last_request_time = current_time
                    return total_waited

                _logger.info(
                    "Rate limiter throttling: sleeping %.2fs (window: %d/%d requests)",
                    wait_time,
                    len(self._timestamps),
                    self.max_rpm,
                )
                await asyncio.sleep(wait_time)
                total_waited += wait_time

    async def __aenter__(self) -> Self:
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        pass


_GLOBAL_RATE_LIMITER: AsyncRateLimiter | None = None


def get_global_rate_limiter(
    max_rpm: int = 12, min_interval_seconds: float = 4.5
) -> AsyncRateLimiter:
    """Retrieve or initialize the global shared rate limiter singleton."""
    global _GLOBAL_RATE_LIMITER
    if _GLOBAL_RATE_LIMITER is None:
        _GLOBAL_RATE_LIMITER = AsyncRateLimiter(
            max_rpm=max_rpm,
            min_interval_seconds=min_interval_seconds,
        )
    return _GLOBAL_RATE_LIMITER
