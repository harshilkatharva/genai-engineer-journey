from __future__ import annotations

from enum import StrEnum


class EventName(StrEnum):
    """Standard observability event names."""

    REQUEST_STARTED = "request_started"
    REQUEST_COMPLETED = "request_completed"
    REQUEST_FAILED = "request_failed"

    QUERY_STARTED = "query_started"
    QUERY_COMPLETED = "query_completed"
    QUERY_FAILED = "query_failed"

    EMBEDDING_STARTED = "embedding_started"
    EMBEDDING_COMPLETED = "embedding_completed"
    EMBEDDING_FAILED = "embedding_failed"

    RETRIEVAL_STARTED = "retrieval_started"
    RETRIEVAL_COMPLETED = "retrieval_completed"
    RETRIEVAL_FAILED = "retrieval_failed"

    PROMPT_STARTED = "prompt_started"
    PROMPT_COMPLETED = "prompt_completed"
    PROMPT_FAILED = "prompt_failed"

    LLM_STARTED = "llm_started"
    LLM_COMPLETED = "llm_completed"
    LLM_FAILED = "llm_failed"
