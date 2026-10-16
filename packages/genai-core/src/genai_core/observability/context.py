from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

_request_id: ContextVar[str | None] = ContextVar("genai_core_request_id", default=None)


def get_request_id() -> str | None:
    """Return the active request ID in current execution context or None."""
    return _request_id.get()


def set_request_id(request_id: str | None = None) -> Token[str | None]:
    """Set the active request ID. If None, auto-generates a UUID4 string."""
    val = request_id if request_id else str(uuid.uuid4())
    return _request_id.set(val)


def reset_request_id(token: Token[str | None]) -> None:
    """Restore the previous request context token."""
    _request_id.reset(token)


@contextmanager
def observability_context(request_id: str | None = None) -> Iterator[str]:
    """Context manager to bind a request_id within a scope and reset on exit."""
    active_id = request_id if request_id else str(uuid.uuid4())
    token = _request_id.set(active_id)
    try:
        yield active_id
    finally:
        _request_id.reset(token)
