"""Unit tests for the evaluation rate limiter."""

from __future__ import annotations

import time

import pytest

from parallel_research_team.eval.rate_limiter import AsyncRateLimiter, get_global_rate_limiter


@pytest.mark.asyncio
async def test_rate_limiter_min_interval_spacing() -> None:
    limiter = AsyncRateLimiter(max_rpm=60, min_interval_seconds=0.1)

    t0 = time.monotonic()
    await limiter.acquire()
    t1 = time.monotonic()
    await limiter.acquire()
    t2 = time.monotonic()

    assert t1 - t0 >= 0.0
    assert t2 - t1 >= 0.09  # Approximately 0.1s delay


@pytest.mark.asyncio
async def test_rate_limiter_context_manager() -> None:
    limiter = AsyncRateLimiter(max_rpm=60, min_interval_seconds=0.05)

    async with limiter:
        pass

    assert len(limiter._timestamps) == 1


def test_get_global_rate_limiter() -> None:
    limiter1 = get_global_rate_limiter()
    limiter2 = get_global_rate_limiter()
    assert limiter1 is limiter2
