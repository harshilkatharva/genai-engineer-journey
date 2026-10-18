from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any

from ..core import get_settings
from ..exceptions import EmbeddingDimensionError, MissingExtraDependencyError

KNOWN_LOCAL_MODEL_DIMENSIONS: dict[str, int] = {
    "all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "paraphrase-MiniLM-L6-v2": 384,
    "sentence-transformers/paraphrase-MiniLM-L6-v2": 384,
    "bge-small-en-v1.5": 384,
    "BAAI/bge-small-en-v1.5": 384,
    "bge-base-en-v1.5": 768,
    "BAAI/bge-base-en-v1.5": 768,
}


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str) -> Any:
    """Load and cache SentenceTransformer model weights in memory."""
    try:
        from sentence_transformers import (  # type: ignore[import-not-found,import-untyped]
            SentenceTransformer,
        )
    except ImportError as err:
        raise MissingExtraDependencyError(extra="local", package="sentence-transformers") from err

    return SentenceTransformer(model_name)


class LocalEmbeddingProvider:
    """Local embedding provider using SentenceTransformers."""

    def __init__(
        self,
        model: str | None = None,
        model_name: str | None = None,
    ) -> None:
        settings = get_settings()
        self.model: str = str(
            model or model_name or getattr(settings, "default_embedding_model", "all-MiniLM-L6-v2")
        )

    @property
    def provider_name(self) -> str:
        return "local"

    @property
    def default_model(self) -> str:
        return "all-MiniLM-L6-v2"

    @property
    def native_dimensions(self) -> int:
        clean_name = self.model.strip()
        if clean_name in KNOWN_LOCAL_MODEL_DIMENSIONS:
            return KNOWN_LOCAL_MODEL_DIMENSIONS[clean_name]

        # Fallback to inspecting loaded model
        model_obj = self._get_model()
        dim = getattr(model_obj, "get_sentence_embedding_dimension", None)
        if callable(dim):
            return int(dim())
        return 384

    def validate_dimensions(self, dimensions: int) -> None:
        expected = self.native_dimensions
        if dimensions != expected:
            raise EmbeddingDimensionError(
                model=self.model,
                requested=dimensions,
                supported=expected,
            )

    def _get_model(self) -> Any:
        return _load_sentence_transformer(self.model)

    async def embed_batch(
        self,
        texts: list[str],
        normalize: bool = True,
    ) -> list[list[float]]:
        if not texts:
            return []

        model = self._get_model()

        # Run CPU-bound PyTorch model inference in a thread to keep the event loop non-blocking
        embeddings = await asyncio.to_thread(
            model.encode,
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )

        return [vector.tolist() for vector in embeddings]
