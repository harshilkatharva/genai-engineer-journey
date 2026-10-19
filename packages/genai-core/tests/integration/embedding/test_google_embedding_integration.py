from __future__ import annotations

import os

import pytest

from genai_core.core import get_settings
from genai_core.embedding import GoogleEmbeddingProvider


def _get_google_api_key() -> str | None:
    return (
        get_settings().google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    )


def _handle_google_error(exc: Exception) -> None:
    err_str = str(exc).lower()
    if any(
        code in err_str
        for code in [
            "404",
            "not found",
            "not_found",
            "429",
            "503",
            "quota",
            "unavailable",
            "high demand",
            "resource_exhausted",
        ]
    ):
        pytest.skip(f"Google API transient rate-limit, quota, or model unavailable: {exc}")
    raise exc


@pytest.mark.integration
@pytest.mark.asyncio
async def test_google_embedding_live() -> None:
    api_key = _get_google_api_key()

    if not api_key:
        pytest.skip("GOOGLE_API_KEY is not set")

    provider = GoogleEmbeddingProvider(
        api_key=api_key, model="gemini-embedding-001", dimensions=768
    )
    try:
        results = await provider.embed_batch(["Hello world from Google embeddings"])
    except Exception as exc:  # noqa: BLE001
        _handle_google_error(exc)
        return

    assert len(results) == 1
    assert len(results[0]) == provider.native_dimensions
