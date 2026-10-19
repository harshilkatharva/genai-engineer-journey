from __future__ import annotations

from ..models.retrieval_model import RetrievalResult


def normalize_min_max(results: list[RetrievalResult]) -> list[RetrievalResult]:
    """Normalize similarity or keyword scores to [0.0, 1.0] using min-max scaling.

    If results list is empty, returns an empty list.
    If all scores are identical, all scores are normalized to 1.0.
    """
    if not results:
        return []

    scores = [r.score for r in results]
    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [
            RetrievalResult(
                id=r.id,
                content=r.content,
                score=1.0,
                rank=r.rank,
                metadata=r.metadata,
                source_id=r.source_id,
            )
            for r in results
        ]

    normalized: list[RetrievalResult] = []
    denom = max_score - min_score
    for r in results:
        norm_score = (r.score - min_score) / denom
        normalized.append(
            RetrievalResult(
                id=r.id,
                content=r.content,
                score=round(norm_score, 6),
                rank=r.rank,
                metadata=r.metadata,
                source_id=r.source_id,
            )
        )
    return normalized


def weighted_score_fusion(
    vector_results: list[RetrievalResult],
    keyword_results: list[RetrievalResult],
    *,
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4,
    top_k: int = 5,
    score_threshold: float | None = None,
) -> list[RetrievalResult]:
    """Fuse dense vector search results and sparse keyword results using weighted linear combination.

    Each input channel is min-max normalized before weighted fusion.
    Candidates present in both channels receive the sum of their weighted scores.
    Candidates present in a single channel receive only that channel's weighted score.
    """
    if vector_weight < 0 or keyword_weight < 0:
        raise ValueError("vector_weight and keyword_weight must be non-negative.")
    if round(vector_weight + keyword_weight, 4) != 1.0:
        raise ValueError(
            f"vector_weight ({vector_weight}) and keyword_weight ({keyword_weight}) must sum to 1.0."
        )
    if top_k <= 0:
        return []

    norm_vector = normalize_min_max(vector_results)
    norm_keyword = normalize_min_max(keyword_results)

    merged: dict[str, tuple[RetrievalResult, float]] = {}

    for vr in norm_vector:
        weighted = vr.score * vector_weight
        merged[vr.id] = (vr, weighted)

    for kr in norm_keyword:
        weighted = kr.score * keyword_weight
        if kr.id in merged:
            prev_result, prev_score = merged[kr.id]
            merged[kr.id] = (prev_result, prev_score + weighted)
        else:
            merged[kr.id] = (kr, weighted)

    # Filter by score_threshold if specified
    candidates: list[tuple[RetrievalResult, float]] = []
    for result, combined_score in merged.values():
        if score_threshold is not None and combined_score < score_threshold:
            continue
        candidates.append((result, combined_score))

    # Sort descending by combined score
    candidates.sort(key=lambda item: item[1], reverse=True)

    # Re-rank up to top_k
    final_results: list[RetrievalResult] = []
    for rank, (result, combined_score) in enumerate(candidates[:top_k], start=1):
        final_results.append(
            RetrievalResult(
                id=result.id,
                content=result.content,
                score=round(combined_score, 6),
                rank=rank,
                metadata=result.metadata,
                source_id=result.source_id,
            )
        )

    return final_results
