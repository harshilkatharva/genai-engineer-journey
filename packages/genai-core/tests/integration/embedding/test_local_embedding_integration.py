from __future__ import annotations

import math

import pytest

from genai_core.embedding import LocalEmbeddingProvider
from genai_core.services import EmbeddingService


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2, strict=True))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_local_embedding_batch_and_dimensions() -> None:
    provider = LocalEmbeddingProvider(model="all-MiniLM-L6-v2")
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming industries worldwide.",
    ]
    embeddings = await provider.embed_batch(texts)

    assert len(embeddings) == 2
    for emb in embeddings:
        assert len(emb) == 384
        # Verify L2 normalization: sum of squares should be very close to 1.0
        norm = sum(x * x for x in emb)
        assert abs(norm - 1.0) < 1e-3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_local_embedding_semantic_similarity() -> None:
    provider = LocalEmbeddingProvider(model="all-MiniLM-L6-v2")
    texts = [
        "The dog is barking loudly in the backyard.",
        "A hound is making noise outside.",
        "Quarterly earnings and financial statements were reported today.",
    ]
    embeddings = await provider.embed_batch(texts)
    dog_emb, hound_emb, finance_emb = embeddings

    sim_dog_hound = _cosine_similarity(dog_emb, hound_emb)
    sim_dog_finance = _cosine_similarity(dog_emb, finance_emb)

    # Dog and hound should be much more similar than dog and finance
    assert sim_dog_hound > sim_dog_finance
    assert sim_dog_hound > 0.40
    assert sim_dog_finance < 0.30


@pytest.mark.integration
@pytest.mark.asyncio
async def test_embedding_service_with_local_provider() -> None:
    provider = LocalEmbeddingProvider(model="all-MiniLM-L6-v2")
    service = EmbeddingService(provider=provider)

    query_emb = await service.embed_query("test query")
    assert len(query_emb) == 384

    doc_embs = await service.embed_documents(["doc one", "doc two"])
    assert len(doc_embs) == 2
    assert len(doc_embs[0]) == 384
    assert len(doc_embs[1]) == 384
