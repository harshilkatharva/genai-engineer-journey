from __future__ import annotations

import pytest

from genai_core.models import RetrievalResult
from genai_core.retrieval.fusion import normalize_min_max, weighted_score_fusion


def _make_result(cid: str, score: float, rank: int = 1) -> RetrievalResult:
    return RetrievalResult(
        id=cid,
        content=f"Content for {cid}",
        score=score,
        rank=rank,
        metadata={"orig": cid},
        source_id="doc1",
    )


def test_normalize_min_max_empty():
    assert normalize_min_max([]) == []


def test_normalize_min_max_single():
    res = [_make_result("c1", 0.75)]
    norm = normalize_min_max(res)
    assert len(norm) == 1
    assert norm[0].score == 1.0
    assert norm[0].id == "c1"


def test_normalize_min_max_identical_scores():
    res = [_make_result("c1", 0.5), _make_result("c2", 0.5), _make_result("c3", 0.5)]
    norm = normalize_min_max(res)
    assert len(norm) == 3
    assert all(r.score == 1.0 for r in norm)


def test_normalize_min_max_spread():
    res = [
        _make_result("c1", 10.0),
        _make_result("c2", 5.0),
        _make_result("c3", 0.0),
    ]
    norm = normalize_min_max(res)
    assert norm[0].score == 1.0
    assert norm[1].score == 0.5
    assert norm[2].score == 0.0


def test_weighted_score_fusion_validation():
    with pytest.raises(ValueError, match="must sum to 1.0"):
        weighted_score_fusion([], [], vector_weight=0.7, keyword_weight=0.7)

    with pytest.raises(ValueError, match="non-negative"):
        weighted_score_fusion([], [], vector_weight=-0.1, keyword_weight=1.1)

    assert weighted_score_fusion([], [], top_k=0) == []


def test_weighted_score_fusion_overlapping_results():
    # c1 is in both, c2 in vector only, c3 in keyword only
    # vector: c1=1.0, c2=0.5 -> norm: c1=1.0, c2=0.0
    # keyword: c1=8.0, c3=4.0 -> norm: c1=1.0, c3=0.0
    v_results = [_make_result("c1", 1.0, 1), _make_result("c2", 0.5, 2)]
    k_results = [_make_result("c1", 8.0, 1), _make_result("c3", 4.0, 2)]

    fused = weighted_score_fusion(
        v_results,
        k_results,
        vector_weight=0.6,
        keyword_weight=0.4,
        top_k=5,
    )

    # c1: 1.0 * 0.6 + 1.0 * 0.4 = 1.0
    # c2: 0.0 * 0.6 + 0.0 = 0.0
    # c3: 0.0 + 0.0 * 0.4 = 0.0
    assert len(fused) == 3
    assert fused[0].id == "c1"
    assert fused[0].score == 1.0
    assert fused[0].rank == 1


def test_weighted_score_fusion_disjoint_results():
    v_results = [_make_result("v1", 0.9), _make_result("v2", 0.1)]
    k_results = [_make_result("k1", 10.0), _make_result("k2", 2.0)]

    # norm v: v1=1.0, v2=0.0
    # norm k: k1=1.0, k2=0.0
    # weights: v=0.7, k=0.3
    # v1 score: 1.0 * 0.7 = 0.7
    # k1 score: 1.0 * 0.3 = 0.3
    # v2 score: 0.0 * 0.7 = 0.0
    # k2 score: 0.0 * 0.3 = 0.0
    fused = weighted_score_fusion(
        v_results,
        k_results,
        vector_weight=0.7,
        keyword_weight=0.3,
        top_k=2,
    )

    assert len(fused) == 2
    assert fused[0].id == "v1"
    assert fused[0].score == 0.7
    assert fused[0].rank == 1
    assert fused[1].id == "k1"
    assert fused[1].score == 0.3
    assert fused[1].rank == 2


def test_weighted_score_fusion_threshold():
    v_results = [_make_result("v1", 1.0), _make_result("v2", 0.5)]
    k_results = [_make_result("k1", 1.0)]

    fused = weighted_score_fusion(
        v_results,
        k_results,
        vector_weight=0.5,
        keyword_weight=0.5,
        top_k=10,
        score_threshold=0.6,
    )
    # Only items with fused score >= 0.6 remain
    assert all(r.score >= 0.6 for r in fused)
