from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from genai_core.exceptions import MissingExtraDependencyError
from genai_core.models import RetrievalResult
from genai_core.retrieval.cross_encoder_reranker import CrossEncoderReranker
from genai_core.retrieval.protocol import Reranker


def _make_result(cid: str, content: str, score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(
        id=cid,
        content=content,
        score=score,
        rank=1,
        metadata={"orig": "meta"},
        source_id="doc1",
    )


def test_cross_encoder_protocol_conformance():
    reranker = CrossEncoderReranker()
    assert isinstance(reranker, Reranker)


@pytest.mark.asyncio
async def test_cross_encoder_empty_results():
    reranker = CrossEncoderReranker()
    res = await reranker.rerank("test query", [], top_k=5)
    assert res == []

    one = [_make_result("c1", "text")]
    res2 = await reranker.rerank("test query", one, top_k=0)
    assert res2 == []


@pytest.mark.asyncio
async def test_cross_encoder_missing_dependency():
    reranker = CrossEncoderReranker()
    with patch.dict(sys.modules, {"sentence_transformers": None}):
        with pytest.raises(MissingExtraDependencyError) as exc_info:
            await reranker.rerank("query", [_make_result("c1", "text")], top_k=5)
        assert exc_info.value.extra == "local"
        assert "sentence-transformers" in exc_info.value.package


@pytest.mark.asyncio
async def test_cross_encoder_rerank_success():
    reranker = CrossEncoderReranker(model_name="mock-model", batch_size=16)

    mock_model = MagicMock()
    # Predict returns raw scores: c2 has highest score, then c1, then c3
    mock_model.predict.return_value = [0.1, 0.9, -0.2]

    results = [
        _make_result("c1", "content 1", score=0.8),
        _make_result("c2", "content 2", score=0.6),
        _make_result("c3", "content 3", score=0.4),
    ]

    with patch.object(reranker, "_load_model", return_value=mock_model):
        reranked = await reranker.rerank("what is ai?", results, top_k=2)

    assert len(reranked) == 2
    # c2 is first
    assert reranked[0].id == "c2"
    assert reranked[0].score == 0.9
    assert reranked[0].rank == 1
    assert reranked[0].metadata["retrieval_score"] == 0.6

    # c1 is second
    assert reranked[1].id == "c1"
    assert reranked[1].score == 0.1
    assert reranked[1].rank == 2
    assert reranked[1].metadata["retrieval_score"] == 0.8

    # Verify model predict called with expected pairs
    mock_model.predict.assert_called_once_with(
        [
            ("what is ai?", "content 1"),
            ("what is ai?", "content 2"),
            ("what is ai?", "content 3"),
        ],
        batch_size=16,
        show_progress_bar=False,
    )
