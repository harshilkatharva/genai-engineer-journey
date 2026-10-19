from __future__ import annotations


class RetrievalError(Exception):
    """Base exception for all retrieval-related errors in genai-core."""

    def __init__(
        self,
        message: str = "An unknown retrieval error occurred.",
        user_message: str | None = None,
    ) -> None:
        self.message = message
        self.user_message = user_message or message
        super().__init__(self.message)


class VectorStoreError(RetrievalError):
    """Raised when an error occurs in the vector store backend or adapter."""

    def __init__(
        self,
        message: str,
        backend: str | None = None,
        retryable: bool = False,
    ) -> None:
        self.backend = backend
        self.retryable = retryable
        prefix = f"[{backend}] " if backend else ""
        super().__init__(message=f"{prefix}{message}")


class IncompatibleEmbeddingError(RetrievalError):
    """Raised when query vector dimension or model does not match store dimension."""

    def __init__(
        self,
        requested: int,
        supported: int,
        message: str | None = None,
    ) -> None:
        self.requested = requested
        self.supported = supported
        msg = message or (
            f"Query embedding dimension ({requested}) does not match "
            f"vector store dimension ({supported})."
        )
        super().__init__(message=msg)


class UnsupportedSearchModeError(RetrievalError):
    """Raised when a requested search mode (e.g. 'sparse', 'hybrid') is not supported."""

    def __init__(self, mode: str, supported_modes: list[str] | None = None) -> None:
        self.mode = mode
        self.supported_modes = supported_modes or ["dense"]
        super().__init__(
            message=(
                f"Search mode '{mode}' is not supported. Supported modes: {self.supported_modes}. "
                "Sparse and hybrid retrieval will be implemented in a future milestone."
            )
        )


class RetrievalTimeoutError(RetrievalError):
    """Raised when a retrieval operation exceeds its configured deadline or timeout."""

    def __init__(self, message: str = "Retrieval operation timed out.") -> None:
        super().__init__(message=message)
