import pytest

from llm_client.exceptions import (
    LLMContentFilterError,
    LLMError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
)


def test_rate_limit_error() -> None:
    with pytest.raises(LLMRateLimitError):
        raise LLMRateLimitError("Rate limit exceeded")


def test_timeout_error() -> None:
    with pytest.raises(LLMTimeoutError):
        raise LLMTimeoutError("Timeout")


def test_content_filter_error() -> None:
    with pytest.raises(LLMContentFilterError):
        raise LLMContentFilterError("Blocked")


def test_invalid_response_error() -> None:
    with pytest.raises(LLMInvalidResponseError):
        raise LLMInvalidResponseError("Invalid JSON")


def test_base_exception() -> None:
    assert issubclass(LLMRateLimitError, LLMError)
