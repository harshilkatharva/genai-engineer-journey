from __future__ import annotations

from enum import StrEnum


class ObservabilityEvent(StrEnum):
    """Standard observability event identifiers across genai-core."""

    LLM_REQUEST_START = "llm_request_start"
    LLM_REQUEST_SUCCESS = "llm_request_success"
    LLM_REQUEST_FAILED = "llm_request_failed"
    LLM_STREAM_START = "llm_stream_start"
    LLM_STREAM_SUCCESS = "llm_stream_success"
    LLM_STREAM_FAILED = "llm_stream_failed"
    LLM_RETRY = "llm_retry"
