"""Rate-limit controller protecting the Module 19 evaluation pipeline against HTTP 429 quota exhaustion.

Handles Google API 15 RPM rate limits by extracting recommended retry delays,
pausing concurrent requests via an async barrier, retrying automatically,
and enforcing strict wall-clock deadlines.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeVar

from genai_core import ObservabilityEvent, safe_log

from self_improving_report_writer.core.config import ReportWriterConfig, get_default_config
from self_improving_report_writer.core.exceptions import EvaluationRateLimitError

if TYPE_CHECKING:
    from genai_core import LLMManagerRequest, LLMManagerResponse, LLMService

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Regex to capture retry delays from provider error messages like:
# - "Please retry in 52.846599s."
# - "Resource has been exhausted. Please retry after 45s"
# - "Rate limit exceeded. retry in 30 seconds"
_RETRY_DELAY_PATTERN = re.compile(
    r"retry\s+(?:after|in)\s+([0-9]+(?:\.[0-9]+)?)\s*s(?:ec(?:ond)?s?)?",
    re.IGNORECASE,
)
_RETRY_DELAY_FALLBACK_PATTERN = re.compile(
    r"retry_delay\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


def is_rate_limit_error(response_or_exc: Any) -> bool:
    """Determine whether a response object or exception indicates an HTTP 429 / RESOURCE_EXHAUSTED rate limit.

    Args:
        response_or_exc: An LLMManagerResponse, Exception, or error mapping.

    Returns:
        True if the target represents an HTTP 429 or quota exhaustion error.
    """
    if response_or_exc is None:
        return False

    # Check LLMManagerResponse or object with .error
    error = getattr(response_or_exc, "error", None)
    target = error if error is not None else response_or_exc

    status_code = getattr(target, "status_code", None)
    if status_code == 429:
        return True

    code = getattr(target, "code", None)
    if code in (429, "429", "RESOURCE_EXHAUSTED", "resource_exhausted"):
        return True
    if code and "429" in str(code):
        return True

    message = getattr(target, "message", None) or str(target)
    message_upper = message.upper()
    if "429" in message_upper:
        return True
    if "RESOURCE_EXHAUSTED" in message_upper:
        return True
    if "RATE LIMIT" in message_upper or "RATELIMIT" in message_upper:
        return True
    if "PLEASE RETRY IN" in message_upper or "PLEASE RETRY AFTER" in message_upper:
        return True

    # Check headers dict if present
    headers = getattr(target, "headers", None) or getattr(response_or_exc, "headers", None)
    return bool(
        isinstance(headers, dict) and ("retry-after" in headers or "Retry-After" in headers)
    )


def extract_retry_delay(
    response_or_exc: Any,
    default_delay: float = 10.0,
) -> float:
    """Extract the recommended retry delay in seconds from an error response, exception, or headers.

    Args:
        response_or_exc: An LLMManagerResponse, Exception, or error mapping.
        default_delay: Fallback duration if no specific delay can be parsed.

    Returns:
        Recommended retry delay in seconds (>= 0.1).
    """
    target = getattr(response_or_exc, "error", None) or response_or_exc

    # 1. Inspect HTTP response headers (e.g., 'Retry-After')
    headers = getattr(target, "headers", None) or getattr(response_or_exc, "headers", None)
    if isinstance(headers, dict):
        for header_name in ("retry-after", "Retry-After", "RETRY-AFTER"):
            if header_name in headers:
                raw_val = headers[header_name]
                try:
                    return max(0.1, float(raw_val))
                except (ValueError, TypeError):
                    pass

    # 2. Inspect error message string
    message = getattr(target, "message", None) or str(target)
    match = _RETRY_DELAY_PATTERN.search(message)
    if match:
        try:
            return max(0.1, float(match.group(1)))
        except (ValueError, IndexError):
            pass

    match_fallback = _RETRY_DELAY_FALLBACK_PATTERN.search(message)
    if match_fallback:
        try:
            return max(0.1, float(match_fallback.group(1)))
        except (ValueError, IndexError):
            pass

    # 3. Check details or body if available
    details = getattr(target, "details", None)
    if isinstance(details, (str, list, dict)):
        details_str = str(details)
        match_details = _RETRY_DELAY_PATTERN.search(details_str)
        if match_details:
            try:
                return max(0.1, float(match_details.group(1)))
            except (ValueError, IndexError):
                pass

    return max(0.1, default_delay)


class EvaluationRateLimitController:
    """Evaluation-scoped rate-limit controller with barrier synchronization and deadline awareness."""

    def __init__(
        self,
        config: ReportWriterConfig | None = None,
        max_retries: int | None = None,
        safety_buffer: float | None = None,
        default_backoff: float | None = None,
    ) -> None:
        cfg = config or get_default_config()
        self.config = cfg
        self.max_retries = (
            max_retries if max_retries is not None else cfg.eval_rate_limit_max_retries
        )
        self.safety_buffer = (
            safety_buffer if safety_buffer is not None else cfg.eval_rate_limit_safety_buffer
        )
        self.default_backoff = (
            default_backoff if default_backoff is not None else cfg.eval_rate_limit_default_backoff
        )

        self._pause_until: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def pause_until(self) -> float:
        """Monotonic timestamp until which requests must pause."""
        return self._pause_until

    @property
    def is_paused(self) -> bool:
        """Whether the controller is currently pausing requests."""
        return time.monotonic() < self._pause_until

    @property
    def pause_remaining(self) -> float:
        """Seconds remaining in current pause window, or 0.0 if not paused."""
        return max(0.0, self._pause_until - time.monotonic())

    async def pause_all(self, delay_seconds: float) -> None:
        """Set or extend the evaluation-wide pause barrier.

        Args:
            delay_seconds: Seconds to pause all subsequent evaluation requests.
        """
        async with self._lock:
            target_time = time.monotonic() + delay_seconds
            if target_time > self._pause_until:
                self._pause_until = target_time
                safe_log(
                    logger,
                    logging.WARNING,
                    f"EvaluationRateLimitController: Pause barrier set for {delay_seconds:.2f}s "
                    f"(until monotonic {self._pause_until:.2f})",
                    event=ObservabilityEvent.LLM_RETRY,
                )

    async def wait_if_paused(self, deadline: float | None = None) -> None:
        """Block if the rate-limit barrier is active, raising an error if waiting would exceed deadline.

        Args:
            deadline: Optional wall-clock monotonic deadline.

        Raises:
            EvaluationRateLimitError: If remaining pause duration exceeds the deadline.
        """
        while True:
            now = time.monotonic()
            remaining = self._pause_until - now
            if remaining <= 0:
                break

            # Check if sleeping would violate the deadline
            if deadline is not None and (now + remaining >= deadline):
                time_left = max(0.0, deadline - now)
                raise EvaluationRateLimitError(
                    f"Evaluation rate-limit pause of {remaining:.2f}s would exceed remaining deadline "
                    f"({time_left:.2f}s left)."
                )

            safe_log(
                logger,
                logging.INFO,
                f"Evaluation rate-limit barrier active: waiting {remaining:.2f}s...",
                event=ObservabilityEvent.LLM_RETRY,
            )
            # Sleep in bounded increments to react to barrier extensions or loop termination
            await asyncio.sleep(min(remaining, 1.0))

    async def execute_with_retry(
        self,
        call_fn: Callable[[], Awaitable[T]],
        deadline: float | None = None,
        max_retries: int | None = None,
    ) -> T:
        """Execute an evaluation LLM call with automatic 429 retry, barrier synchronization, and deadline checks.

        Args:
            call_fn: Async callable returning the LLM response.
            deadline: Optional monotonic wall-clock deadline.
            max_retries: Optional override for maximum retry attempts.

        Returns:
            The successful result of call_fn.

        Raises:
            EvaluationRateLimitError: If retries are exhausted or pause would exceed deadline.
            Exception: Any non-retryable provider exception.
        """
        retries = max_retries if max_retries is not None else self.max_retries

        for attempt in range(retries + 1):
            # 1. Wait if barrier is currently active
            await self.wait_if_paused(deadline=deadline)

            # 2. Check deadline before initiating call
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                raise EvaluationRateLimitError(
                    f"Evaluation deadline exceeded before LLM call execution (attempt {attempt + 1})."
                )

            # 3. Execute the call
            try:
                result = await call_fn()
            except Exception as exc:
                if is_rate_limit_error(exc) and attempt < retries:
                    parsed_delay = extract_retry_delay(
                        exc, default_delay=self.default_backoff * (2**attempt)
                    )
                    effective_delay = parsed_delay + self.safety_buffer

                    now = time.monotonic()
                    if deadline is not None and (now + effective_delay >= deadline):
                        time_left = max(0.0, deadline - now)
                        raise EvaluationRateLimitError(
                            f"HTTP 429 rate limit encountered on attempt {attempt + 1}. "
                            f"Required delay of {effective_delay:.2f}s exceeds remaining deadline ({time_left:.2f}s)."
                        ) from exc

                    safe_log(
                        logger,
                        logging.WARNING,
                        f"HTTP 429 rate limit exception on attempt {attempt + 1}/{retries + 1}. "
                        f"Pausing for {effective_delay:.2f}s (parsed={parsed_delay:.2f}s + buffer={self.safety_buffer:.2f}s)...",
                        event=ObservabilityEvent.LLM_RETRY,
                    )
                    await self.pause_all(effective_delay)
                    continue
                raise

            # 4. Check if structured response returned an error indicating 429
            if is_rate_limit_error(result) and attempt < retries:
                parsed_delay = extract_retry_delay(
                    result, default_delay=self.default_backoff * (2**attempt)
                )
                effective_delay = parsed_delay + self.safety_buffer

                now = time.monotonic()
                if deadline is not None and (now + effective_delay >= deadline):
                    time_left = max(0.0, deadline - now)
                    safe_log(
                        logger,
                        logging.WARNING,
                        f"HTTP 429 rate limit in response on attempt {attempt + 1}. "
                        f"Delay {effective_delay:.2f}s exceeds remaining deadline ({time_left:.2f}s). Aborting retries.",
                        event=ObservabilityEvent.LLM_REQUEST_FAILED,
                    )
                    # Return the error response to let caller handle its terminal outcome
                    return result

                safe_log(
                    logger,
                    logging.WARNING,
                    f"HTTP 429 rate limit in response on attempt {attempt + 1}/{retries + 1}. "
                    f"Pausing for {effective_delay:.2f}s (parsed={parsed_delay:.2f}s + buffer={self.safety_buffer:.2f}s)...",
                    event=ObservabilityEvent.LLM_RETRY,
                )
                await self.pause_all(effective_delay)
                continue

            return result

        # Exhausted retries
        return result

    async def complete_structure(
        self,
        llm_service: LLMService,
        prompt: str,
        response_schema: type[T],
        deadline: float | None = None,
        **kwargs: Any,
    ) -> LLMManagerResponse:
        """Call LLMService.complete_structure with rate-limit and deadline management."""

        async def _call() -> LLMManagerResponse:
            return await llm_service.complete_structure(
                prompt=prompt,
                response_schema=response_schema,
                deadline=deadline,
                **kwargs,
            )

        return await self.execute_with_retry(_call, deadline=deadline)

    async def complete(
        self,
        llm_service: LLMService,
        request: LLMManagerRequest,
        deadline: float | None = None,
    ) -> LLMManagerResponse:
        """Call LLMService.complete with rate-limit and deadline management."""

        async def _call() -> LLMManagerResponse:
            return await llm_service.complete(request)

        return await self.execute_with_retry(_call, deadline=deadline)

    def reset(self) -> None:
        """Reset the pause barrier and active delays (for testing or run isolation)."""
        self._pause_until = 0.0


# Module-level shared singleton for evaluation components
_SHARED_CONTROLLER: EvaluationRateLimitController | None = None


def get_shared_rate_limit_controller(
    config: ReportWriterConfig | None = None,
) -> EvaluationRateLimitController:
    """Obtain or initialize the process-wide EvaluationRateLimitController."""
    global _SHARED_CONTROLLER
    if _SHARED_CONTROLLER is None:
        _SHARED_CONTROLLER = EvaluationRateLimitController(config=config)
    return _SHARED_CONTROLLER


def reset_shared_rate_limit_controller() -> None:
    """Reset the module-level shared evaluation rate limit controller."""
    global _SHARED_CONTROLLER
    if _SHARED_CONTROLLER is not None:
        _SHARED_CONTROLLER.reset()
    _SHARED_CONTROLLER = None
