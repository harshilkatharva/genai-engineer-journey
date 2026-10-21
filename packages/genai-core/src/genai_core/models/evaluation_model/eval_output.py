from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class EvalOutput(BaseModel):
    """Standardized output emitted by the system under test (live or pre-recorded)."""

    answer: str | None = None
    retrieved_chunk_ids: list[str] | None = None
    retrieved_contexts: list[str] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    latency_ms: float = 0.0
    token_usage: dict[str, int] | None = None
    error: str | None = None
    raw_response: Any = None
