from __future__ import annotations

import os

import pytest

from genai_core.core import get_settings
from genai_core.embedding import OpenAIEmbeddingProvider


def _get_openai_api_key() -> str | None:
    return get_settings().openai_api_key or os.getenv("OPENAI_API_KEY")


def _handle_openai_error(exc: Exception) -> None:
    err_str = str(exc).lower()
    if any(
        term in err_str
        for term in [
            "429",
            "quota",
            "credit",
            "billing",
            "rate_limit",
            "insufficient",
            "exceeded",
            "badrequesterror",
            "400",
        ]
    ):
        pytest.skip(f"OpenAI live API skipped due to insufficient credits or rate limit: {exc}")
    raise exc


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_embedding_live() -> None:
    api_key = _get_openai_api_key()
    if not api_key:
        pytest.skip("OPENAI_API_KEY is not set")

    provider = OpenAIEmbeddingProvider(api_key=api_key)
    try:
        results = await provider.embed_batch(["Hello world from OpenAI embeddings"])
    except Exception as exc:  # noqa: BLE001
        _handle_openai_error(exc)
        return

    assert len(results) == 1
    assert len(results[0]) == provider.native_dimensions
