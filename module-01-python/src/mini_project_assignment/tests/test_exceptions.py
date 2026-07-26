import pytest

from mini_project_assignment.exceptions import (
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMContentFilterError,
    LLMInvalidResponseError,
)


def test_rate_limit_error():

    with pytest.raises(LLMRateLimitError):
        raise LLMRateLimitError("Rate limit exceeded")


def test_timeout_error():

    with pytest.raises(LLMTimeoutError):
        raise LLMTimeoutError("Timeout")


def test_content_filter_error():

    with pytest.raises(LLMContentFilterError):
        raise LLMContentFilterError("Blocked")


def test_invalid_response_error():

    with pytest.raises(LLMInvalidResponseError):
        raise LLMInvalidResponseError("Invalid JSON")


def test_base_exception():

    assert issubclass(LLMRateLimitError, LLMError)