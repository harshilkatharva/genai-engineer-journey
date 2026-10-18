from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol that all local and hosted embedding adapters must satisfy."""

    @property
    def provider_name(self) -> str:
        """Return the unique identifier for this provider (e.g. 'local', 'google', 'openai')."""
        ...

    @property
    def default_model(self) -> str:
        """Return the default embedding model for this provider."""
        ...

    @property
    def native_dimensions(self) -> int:
        """Return the native vector dimension produced by the model."""
        ...

    def validate_dimensions(self, dimensions: int) -> None:
        """Validate whether the configured output dimension is supported.

        Raises EmbeddingDimensionError if unsupported.
        """
        ...

    async def embed_batch(
        self,
        texts: list[str],
        normalize: bool = True,
    ) -> list[list[float]]:
        """Vectorize a batch of text strings into float embeddings."""
        ...
