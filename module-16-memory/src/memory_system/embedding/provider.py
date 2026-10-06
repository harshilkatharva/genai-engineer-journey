from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol

from ..core import get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingProvider(Protocol):
    # Converts text into a numeric vector suitable for semantic search.
    async def embed(self, text: str) -> list[float]: ...


class SentenceTransformerEmbeddings:
    # Configures the local sentence-transformer model and expected vector size.
    def __init__(self, model_name: str | None = None, dimension: int | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.memory_embedding_model
        self.dimension = dimension or settings.memory_embedding_dimension
        self._model: SentenceTransformer | None = None

    # Produces a normalized embedding without blocking the event loop.
    async def embed(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Cannot embed empty text")
        return await asyncio.to_thread(self._embed_sync, text)

    # Loads the configured model once and generates a validated text embedding.
    def _embed_sync(self, text: str) -> list[float]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        vector = self._model.encode(text, normalize_embeddings=True).tolist()
        if len(vector) != self.dimension:
            raise ValueError(
                f"Embedding model returned {len(vector)} dimensions; configured for "
                f"{self.dimension}"
            )
        return [float(value) for value in vector]
