from __future__ import annotations

import math
from typing import Any

from google import genai
from google.genai import types

from ..core import get_settings
from ..exceptions.embedding_exceptions import (
    EmbeddingDimensionError,
    EmbeddingProviderError,
)

GOOGLE_KNOWN_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-005": 768,
    "models/text-embedding-005": 768,
    "embedding-001": 768,
    "models/embedding-001": 768,
}


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return vector
    return [x / norm for x in vector]


class GoogleEmbeddingProvider:
    """Hosted Google Gemini embedding provider adapter using google-genai SDK."""

    def __init__(
        self,
        client: genai.Client | None = None,
        model: str | None = None,
        api_key: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        settings = get_settings()
        effective_key = api_key or settings.google_api_key
        try:
            self.client = client or (
                genai.Client(api_key=effective_key) if effective_key else genai.Client()
            )
        except Exception as err:
            raise EmbeddingProviderError(
                "google",
                f"Google API key missing or invalid. Set GEMINI_API_KEY environment variable or settings.google_api_key: {err}",
            ) from err

        self.model: str = str(
            model or getattr(settings, "default_google_embedding_model", "text-embedding-005")
        )
        self.dimensions: int | None = dimensions
        self.last_usage: dict[str, int] | None = None

    @property
    def provider_name(self) -> str:
        return "google"

    @property
    def default_model(self) -> str:
        return "text-embedding-005"

    @property
    def native_dimensions(self) -> int:
        clean_model = self.model.strip()
        return GOOGLE_KNOWN_MODEL_DIMENSIONS.get(clean_model, 768)

    def validate_dimensions(self, dimensions: int) -> None:
        max_dim = self.native_dimensions
        if dimensions <= 0 or dimensions > max_dim:
            raise EmbeddingDimensionError(
                model=self.model,
                requested=dimensions,
                supported=max_dim,
            )
        self.dimensions = dimensions

    async def embed_batch(
        self,
        texts: list[str],
        normalize: bool = True,
    ) -> list[list[float]]:
        if not texts:
            return []

        config_kwargs: dict[str, Any] = {}
        if self.dimensions is not None:
            config_kwargs["output_dimensionality"] = self.dimensions

        config = types.EmbedContentConfig(**config_kwargs) if config_kwargs else None

        try:
            resp = await self.client.aio.models.embed_content(
                model=self.model,
                contents=texts,
                config=config,
            )
        except Exception as err:
            raise EmbeddingProviderError("google", str(err)) from err

        embeddings: list[list[float]] = []
        raw_embeddings = getattr(resp, "embeddings", None)
        if raw_embeddings is not None:
            for emb in raw_embeddings:
                raw_values = getattr(emb, "values", None)
                if raw_values is not None:
                    embeddings.append([float(val) for val in raw_values])
        else:
            single_emb = getattr(resp, "embedding", None)
            if single_emb is not None:
                raw_values = getattr(single_emb, "values", None)
                if raw_values is not None:
                    embeddings.append([float(val) for val in raw_values])

        if len(embeddings) != len(texts):
            raise EmbeddingProviderError(
                "google",
                f"Google API returned {len(embeddings)} vectors for batch of {len(texts)} texts.",
            )

        if normalize:
            embeddings = [_l2_normalize(vec) for vec in embeddings]

        return embeddings
