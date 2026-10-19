from __future__ import annotations

import asyncio
from typing import Any

from ..core import get_settings
from ..exceptions import MissingExtraDependencyError
from ..models.retrieval_model import RetrievalResult


class CrossEncoderReranker:
    """Re-ranks retrieved candidates using a cross-encoder model.

    Cross-encoders attend jointly over the (query, document) pair, yielding
    finer-grained relevance scoring than bi-encoder embeddings alone.
    """

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        max_length: int = 512,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model_name
        self.batch_size = batch_size or settings.reranker_batch_size
        self.max_length = max_length
        self._model: Any = None

    def _load_model(self) -> Any:
        """Lazily load the sentence-transformers CrossEncoder model."""
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
        except ImportError as err:
            raise MissingExtraDependencyError(
                extra="local",
                package="sentence-transformers",
                feature="CrossEncoderReranker",
            ) from err

        self._model = CrossEncoder(
            self.model_name,
            max_length=self.max_length,
        )
        return self._model

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Re-rank candidate results against the query and return the top_k.

        The original retrieval score is preserved in metadata['retrieval_score'].
        """
        if not results or top_k <= 0:
            return []

        model = self._load_model()
        pairs = [(query, result.content) for result in results]

        raw_scores = await asyncio.to_thread(
            model.predict,
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        scored_candidates: list[tuple[RetrievalResult, float]] = []
        for result, score in zip(results, raw_scores, strict=True):
            updated_meta = dict(result.metadata)
            updated_meta["retrieval_score"] = result.score
            scored_candidates.append(
                (
                    RetrievalResult(
                        id=result.id,
                        content=result.content,
                        score=round(float(score), 6),
                        rank=result.rank,
                        metadata=updated_meta,
                        source_id=result.source_id,
                    ),
                    float(score),
                )
            )

        # Sort descending by cross-encoder score
        scored_candidates.sort(key=lambda item: item[1], reverse=True)

        final_results: list[RetrievalResult] = []
        for rank, (result, _) in enumerate(scored_candidates[:top_k], start=1):
            result.rank = rank
            final_results.append(result)

        return final_results
