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

    # Embedding events
    EMBEDDING_REQUEST_START = "embedding_request_start"
    EMBEDDING_REQUEST_SUCCESS = "embedding_request_success"
    EMBEDDING_REQUEST_FAILED = "embedding_request_failed"
    EMBEDDING_RATE_LIMITED = "embedding_rate_limited"
    EMBEDDING_RETRY = "embedding_retry"
    EMBEDDING_TIMEOUT = "embedding_timeout"

    # Retrieval events
    RETRIEVAL_START = "retrieval_start"
    RETRIEVAL_SUCCESS = "retrieval_success"
    RETRIEVAL_FAILED = "retrieval_failed"
    RETRIEVAL_INDEX_START = "retrieval_index_start"
    RETRIEVAL_INDEX_SUCCESS = "retrieval_index_success"
    RETRIEVAL_DELETE_SUCCESS = "retrieval_delete_success"
    RETRIEVAL_RERANK_START = "retrieval_rerank_start"
    RETRIEVAL_RERANK_SUCCESS = "retrieval_rerank_success"
    RETRIEVAL_RERANK_SKIPPED = "retrieval_rerank_skipped"
    RETRIEVAL_RERANK_FAILED = "retrieval_rerank_failed"

    # Memory events
    MEMORY_STORE_SUCCESS = "memory_store_success"
    MEMORY_RETRIEVE_SUCCESS = "memory_retrieve_success"
    MEMORY_DELETE_SUCCESS = "memory_delete_success"
    MEMORY_PURGE_SUCCESS = "memory_purge_success"
    MEMORY_ACTION_CHECK = "memory_action_check"
    MEMORY_CONSOLIDATION_START = "memory_consolidation_start"
    MEMORY_CONSOLIDATION_SUCCESS = "memory_consolidation_success"
    MEMORY_CONSOLIDATION_FAILED = "memory_consolidation_failed"

    # Tool events
    TOOL_EXECUTION_START = "tool_execution_start"
    TOOL_EXECUTION_SUCCESS = "tool_execution_success"
    TOOL_EXECUTION_FAILED = "tool_execution_failed"
    TOOL_APPROVAL_REQUIRED = "tool_approval_required"
    TOOL_APPROVAL_DECISION = "tool_approval_decision"
    TOOL_LOOP_DETECTED = "tool_loop_detected"
    TOOL_LOOP_START = "tool_loop_start"
    TOOL_LOOP_STEP = "tool_loop_step"
    TOOL_LOOP_COMPLETED = "tool_loop_completed"
