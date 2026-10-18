from __future__ import annotations

import math
from typing import Any

from openai import AsyncOpenAI

from ..core import get_settings
from ..exceptions.embedding_exceptions import (
    EmbeddingDimensionError,
    EmbeddingProviderError,
)

OPENAI_KNOWN_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return vector
    return [x / norm for x in vector]


class OpenAIEmbeddingProvider:
    """Hosted OpenAI embedding provider adapter using AsyncOpenAI."""

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
        api_key: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        settings = get_settings()
        effective_key = api_key or settings.openai_api_key
        try:
            self.client = client or (
                AsyncOpenAI(api_key=effective_key) if effective_key else AsyncOpenAI()
            )
        except Exception as err:
            raise EmbeddingProviderError(
                "openai",
                f"OpenAI API key missing or invalid. Set OPENAI_API_KEY environment variable or settings.openai_api_key: {err}",
            ) from err

        self.model: str = str(
            model or getattr(settings, "default_openai_embedding_model", "text-embedding-3-small")
        )
        self.dimensions: int | None = dimensions
        self.last_usage: dict[str, int] | None = None

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return "text-embedding-3-small"

    @property
    def native_dimensions(self) -> int:
        clean_model = self.model.strip()
        return OPENAI_KNOWN_MODEL_DIMENSIONS.get(clean_model, 1536)

    def validate_dimensions(self, dimensions: int) -> None:
        clean_model = self.model.strip()
        max_dim = self.native_dimensions

        if clean_model == "text-embedding-ada-002":
            if dimensions != 1536:
                raise EmbeddingDimensionError(
                    model=self.model,
                    requested=dimensions,
                    supported=1536,
                )
        else:
            # text-embedding-3-* models support reduced output dimensions up to native max
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

        clean_model = self.model.strip()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }

        # text-embedding-3-* supports native output dimension truncation parameter
        if self.dimensions is not None and clean_model != "text-embedding-ada-002":
            kwargs["dimensions"] = self.dimensions

        try:
            resp = await self.client.embeddings.create(**kwargs)
        except Exception as err:
            raise EmbeddingProviderError("openai", str(err)) from err

        # Track usage metadata
        if getattr(resp, "usage", None):
            self.last_usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "total_tokens": resp.usage.total_tokens,
            }

        # OpenAI embeddings API items have index and embedding
        ordered_data = sorted(resp.data, key=lambda item: item.index)
        embeddings = [list(item.embedding) for item in ordered_data]

        if normalize:
            embeddings = [_l2_normalize(vec) for vec in embeddings]

        return embeddings
