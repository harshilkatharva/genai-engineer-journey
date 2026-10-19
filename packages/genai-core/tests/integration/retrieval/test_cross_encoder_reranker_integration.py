from __future__ import annotations

import pytest

from genai_core.models import RetrievalResult
from genai_core.retrieval import CrossEncoderReranker


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cross_encoder_reranker_live_scoring() -> None:
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L6-v2")

    candidates = [
        RetrievalResult(
            id="doc1",
            content="The Apollo program was an American human spaceflight program carried out by NASA.",
            score=0.75,
            rank=1,
            metadata={"source": "nasa"},
        ),
        RetrievalResult(
            id="doc2",
            content="French cuisine is known for its fine wines, cheeses, baguettes, and croissants.",
            score=0.70,
            rank=2,
            metadata={"source": "cooking"},
        ),
        RetrievalResult(
            id="doc3",
            content="NASA landed the first humans on the Moon during the Apollo 11 mission in July 1969.",
            score=0.65,
            rank=3,
            metadata={"source": "moon_landing"},
        ),
    ]

    query = "When did astronauts land on the moon?"
    ranked = await reranker.rerank(query, candidates, top_k=3)

    assert len(ranked) == 3
    # doc3 specifically answers "when" and "landed on the Moon"
    assert ranked[0].id == "doc3"
    assert ranked[1].id == "doc1"
    assert ranked[2].id == "doc2"

    # Verify ranks are updated sequentially
    assert ranked[0].rank == 1
    assert ranked[1].rank == 2
    assert ranked[2].rank == 3

    # Verify original scores are recorded in metadata
    assert ranked[0].metadata["retrieval_score"] == 0.65
    assert ranked[1].metadata["retrieval_score"] == 0.75
    assert ranked[2].metadata["retrieval_score"] == 0.70

    # The relevant document score must be strictly greater than distractor
    assert ranked[0].score > ranked[2].score


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cross_encoder_reranker_top_k_limiting() -> None:
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L6-v2")
    candidates = [
        RetrievalResult(id=f"doc_{i}", content=f"Document text number {i}", score=float(i), rank=i)
        for i in range(1, 6)
    ]
    ranked = await reranker.rerank("Document text", candidates, top_k=2)
    assert len(ranked) == 2
    assert ranked[0].rank == 1
    assert ranked[1].rank == 2
