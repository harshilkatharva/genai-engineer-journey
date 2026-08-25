from unittest.mock import MagicMock, patch

import pytest

from rag_app.models import RetriveResult
from rag_app.retrieval.re_ranker.cross_encoder_reranker import (
    CrossEncoderReranker,
)


@pytest.fixture
def mock_settings():
    settings = MagicMock()

    settings.re_ranker_model_name = "test-cross-encoder"
    settings.re_ranker_max_length = 512
    settings.re_ranker_batch_size = 8

    return settings


@pytest.fixture
def reranker(mock_settings):
    fake_model = MagicMock()

    with (
        patch(
            "rag_app.retrieval.re_ranker.cross_encoder_reranker.get_settings",
            return_value=mock_settings,
        ),
        patch(
            "rag_app.retrieval.re_ranker.cross_encoder_reranker.get_model",
            return_value=fake_model,
        ),
    ):
        reranker = CrossEncoderReranker()

    reranker.model = fake_model

    return reranker


@pytest.mark.asyncio
async def test_rerank_returns_empty_when_results_are_empty(
    reranker,
):
    results = await reranker.rerank(
        queries=["employee leave policy"],
        results=[],
        top_k=5,
    )

    assert results == []


@pytest.mark.asyncio
async def test_rerank_builds_query_chunk_pairs(
    reranker,
):
    reranker.model = MagicMock()

    reranker.model.predict.return_value = [
        0.90,
        0.40,
        0.60,
        0.80,
    ]

    results = [
        RetriveResult(
            chunk_id="chunk_1",
            chunk_text="Employee leave policy",
            similarity_score=0.0,
        ),
        RetriveResult(
            chunk_id="chunk_2",
            chunk_text="HR vacation process",
            similarity_score=0.0,
        ),
    ]

    queries = [
        "employee leave",
        "annual vacation",
    ]

    await reranker.rerank(
        queries=queries,
        results=results,
        top_k=2,
    )

    reranker.model.predict.assert_called_once()

    pairs = reranker.model.predict.call_args.args[0]

    assert pairs == [
        ("employee leave", "Employee leave policy"),
        ("employee leave", "HR vacation process"),
        ("annual vacation", "Employee leave policy"),
        ("annual vacation", "HR vacation process"),
    ]


@pytest.mark.asyncio
async def test_rerank_keeps_highest_score_for_each_chunk(
    reranker,
):
    reranker.model = MagicMock()

    # Pair order:
    #
    # query_1 + chunk_1 -> 0.50
    # query_1 + chunk_2 -> 0.40
    # query_2 + chunk_1 -> 0.95
    # query_2 + chunk_2 -> 0.60

    reranker.model.predict.return_value = [
        0.50,
        0.40,
        0.95,
        0.60,
    ]

    results = [
        RetriveResult(
            chunk_id="chunk_1",
            chunk_text="Employee leave policy",
            similarity_score=0.0,
        ),
        RetriveResult(
            chunk_id="chunk_2",
            chunk_text="HR vacation process",
            similarity_score=0.0,
        ),
    ]

    reranked = await reranker.rerank(
        queries=[
            "employee leave",
            "annual vacation",
        ],
        results=results,
        top_k=2,
    )

    assert len(reranked) == 2

    assert reranked[0].chunk_id == "chunk_1"
    assert reranked[0].similarity_score == pytest.approx(0.95)

    assert reranked[1].chunk_id == "chunk_2"
    assert reranked[1].similarity_score == pytest.approx(0.60)


@pytest.mark.asyncio
async def test_rerank_sorts_results_by_score(
    reranker,
):
    reranker.model = MagicMock()

    reranker.model.predict.return_value = [
        0.30,
        0.90,
        0.60,
    ]

    results = [
        RetriveResult(
            chunk_id="chunk_1",
            chunk_text="Chunk 1",
            similarity_score=0.0,
        ),
        RetriveResult(
            chunk_id="chunk_2",
            chunk_text="Chunk 2",
            similarity_score=0.0,
        ),
        RetriveResult(
            chunk_id="chunk_3",
            chunk_text="Chunk 3",
            similarity_score=0.0,
        ),
    ]

    reranked = await reranker.rerank(
        queries=["test query"],
        results=results,
        top_k=3,
    )

    assert [result.chunk_id for result in reranked] == [
        "chunk_2",
        "chunk_3",
        "chunk_1",
    ]

    assert [result.similarity_score for result in reranked] == [
        pytest.approx(0.90),
        pytest.approx(0.60),
        pytest.approx(0.30),
    ]


@pytest.mark.asyncio
async def test_rerank_respects_top_k(
    reranker,
):
    reranker.model = MagicMock()

    reranker.model.predict.return_value = [
        0.30,
        0.90,
        0.60,
        0.80,
    ]

    results = [
        RetriveResult(
            chunk_id="chunk_1",
            chunk_text="Chunk 1",
            similarity_score=0.0,
        ),
        RetriveResult(
            chunk_id="chunk_2",
            chunk_text="Chunk 2",
            similarity_score=0.0,
        ),
        RetriveResult(
            chunk_id="chunk_3",
            chunk_text="Chunk 3",
            similarity_score=0.0,
        ),
        RetriveResult(
            chunk_id="chunk_4",
            chunk_text="Chunk 4",
            similarity_score=0.0,
        ),
    ]

    reranked = await reranker.rerank(
        queries=["test query"],
        results=results,
        top_k=2,
    )

    assert len(reranked) == 2

    assert [result.chunk_id for result in reranked] == [
        "chunk_2",
        "chunk_4",
    ]


@pytest.mark.asyncio
async def test_rerank_uses_configured_batch_size(
    reranker,
):
    reranker.model = MagicMock()

    reranker.model.predict.return_value = [0.80]

    results = [
        RetriveResult(
            chunk_id="chunk_1",
            chunk_text="Employee policy",
            similarity_score=0.0,
        )
    ]

    await reranker.rerank(
        queries=["employee policy"],
        results=results,
        top_k=1,
    )

    reranker.model.predict.assert_called_once_with(
        [("employee policy", "Employee policy")],
        batch_size=reranker.settings.re_ranker_batch_size,
        show_progress_bar=False,
    )


@pytest.mark.asyncio
async def test_rerank_preserves_chunk_data(
    reranker,
):
    reranker.model = MagicMock()

    reranker.model.predict.return_value = [0.91]

    result = RetriveResult(
        chunk_id="chunk_123",
        chunk_text="Employee password reset procedure.",
        similarity_score=0.42,
    )

    reranked = await reranker.rerank(
        queries=["password reset"],
        results=[result],
        top_k=1,
    )

    assert reranked[0].chunk_id == "chunk_123"
    assert reranked[0].chunk_text == ("Employee password reset procedure.")
    assert reranked[0].similarity_score == pytest.approx(0.91)
