from __future__ import annotations

from contextvars import ContextVar, Token


# Current request ID for the active async execution context.
_request_id: ContextVar[str | None] = ContextVar(
    "request_id",
    default=None,
)


def set_request_id(request_id: str) -> Token[str | None]:
    """
    Set the request ID for the current execution context.

    Returns a token that can be used to restore the
    previous context value.
    """
    return _request_id.set(request_id)


def get_request_id() -> str | None:
    """Return the current request ID."""
    return _request_id.get()


def reset_request_id(token: Token[str | None]) -> None:
    """Restore the previous request context."""
    _request_id.reset(token)
