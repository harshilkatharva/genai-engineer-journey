from __future__ import annotations


class EmbeddingError(Exception):
    """Base exception for all embedding operations."""

    def __init__(
        self,
        message: str = "An unknown embedding error occurred.",
        user_message: str | None = None,
    ) -> None:
        self.message = message
        self.user_message = user_message or message
        super().__init__(self.message)


class EmbeddingDimensionError(EmbeddingError):
    """Raised when a requested embedding dimension is unsupported by the model."""

    def __init__(self, model: str, requested: int, supported: int | list[int]) -> None:
        self.model = model
        self.requested = requested
        self.supported = supported
        message = (
            f"Model '{model}' does not support output dimension {requested}. "
            f"Supported dimension(s): {supported}."
        )
        super().__init__(message=message)


class MissingExtraDependencyError(EmbeddingError):
    """Raised when a required optional dependency extra is not installed."""

    def __init__(self, extra: str = "local", package: str = "sentence-transformers") -> None:
        self.extra = extra
        self.package = package
        message = (
            f"The '{extra}' embedding provider requires '{package}'. "
            f"Install it in your active environment using: "
            f"pip install -e 'packages/genai-core[{extra}]' "
            f"or uv pip install -e 'packages/genai-core[{extra}]'"
        )
        super().__init__(message=message)


class EmbeddingProviderError(EmbeddingError):
    """Raised when an embedding provider fails or an unsupported provider is requested."""

    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        super().__init__(message=f"Embedding provider '{provider}' error: {message}")


class EmbeddingTimeoutError(EmbeddingError):
    """Raised when an embedding request or sub-batch exceeds its timeout deadline."""

    def __init__(self, message: str = "Embedding request timed out.") -> None:
        super().__init__(message=message)


class EmbeddingRateLimitError(EmbeddingError):
    """Raised when an embedding request exceeds rate limits and cannot be fulfilled within bounds."""

    def __init__(
        self,
        provider: str,
        message: str = "Rate limit exceeded.",
        retry_after: float | None = None,
    ) -> None:
        self.provider = provider
        self.retry_after = retry_after
        detail = f" (retry after {retry_after}s)" if retry_after is not None else ""
        super().__init__(
            message=f"Embedding rate limit for '{provider}' exceeded{detail}: {message}"
        )
