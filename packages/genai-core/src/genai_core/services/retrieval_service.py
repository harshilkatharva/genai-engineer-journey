import asyncio
import time
import uuid
from typing import Any

from ..core import get_settings
from ..exceptions import (
    EmbeddingTimeoutError,
    IncompatibleEmbeddingError,
    MissingExtraDependencyError,
    RetrievalError,
    RetrievalTimeoutError,
    UnsupportedSearchModeError,
)
from ..models.retrieval_model import (
    DocumentChunk,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalResult,
)
from ..observability import ObservabilityEvent, get_logger, safe_log
from ..resilience import RetryPolicy, execute_with_retry
from ..retrieval.fusion import weighted_score_fusion
from ..retrieval.protocol import Reranker, VectorStoreAdapter
from .embedding_service import EmbeddingService

_logger = get_logger("genai_core.services.retrieval_service")


class RetrievalService:
    """Orchestrates query validation, embedding, resilient vector/keyword search, fusion, and re-ranking."""

    def __init__(
        self,
        store: VectorStoreAdapter,
        embedding_service: EmbeddingService | None = None,
        reranker: Reranker | None = None,
        enable_reranking: bool | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.store = store
        self.embedding_service = embedding_service
        self.reranker = reranker

        settings = get_settings()
        self.enable_reranking = (
            enable_reranking if enable_reranking is not None else settings.enable_reranking
        )

        # Strict validation of dimension compatibility between embedding service and store
        if (
            self.embedding_service is not None
            and self.embedding_service.dimensions != self.store.dimensions
        ):
            raise IncompatibleEmbeddingError(
                requested=self.embedding_service.dimensions,
                supported=self.store.dimensions,
                message=(
                    f"EmbeddingService dimension ({self.embedding_service.dimensions}) does not "
                    f"match VectorStore dimension ({self.store.dimensions})."
                ),
            )

        self.retry_policy = retry_policy or RetryPolicy(
            max_retries=2,
            initial_backoff=0.2,
            max_backoff=5.0,
            timeout_seconds=settings.retrieval_timeout_seconds,
        )

    async def search(
        self,
        query: RetrievalQuery | str,
        **kwargs: Any,
    ) -> RetrievalResponse:
        """Execute semantic, keyword, or hybrid search against the configured vector store.

        Accepts either a structured RetrievalQuery or a string query with keyword overrides.
        """
        start_time = time.perf_counter()
        settings = get_settings()

        if isinstance(query, str):
            req = RetrievalQuery(query=query, **kwargs)
        elif isinstance(query, RetrievalQuery):
            req = query
        else:
            raise TypeError(
                f"Expected query to be a str or RetrievalQuery, got {type(query).__name__}."
            )

        if req.search_mode not in ("dense", "sparse", "hybrid"):
            raise UnsupportedSearchModeError(
                req.search_mode, supported_modes=["dense", "sparse", "hybrid"]
            )

        should_rerank = req.rerank if req.rerank is not None else self.enable_reranking
        final_top_k = req.rerank_top_k or req.top_k

        # Determine candidate pool size (retrieve wider candidate pool if re-ranking or fusing)
        if should_rerank:
            candidate_top_k = max(req.top_k * 3, 20, final_top_k * 2)
        elif req.search_mode == "hybrid":
            candidate_top_k = max(req.top_k * 2, req.top_k)
        else:
            candidate_top_k = req.top_k

        # Resolve query embedding vector if required for dense or hybrid modes
        query_vector: list[float] | None = None
        if req.search_mode in ("dense", "hybrid"):
            if req.query_vector is not None:
                if len(req.query_vector) != self.store.dimensions:
                    raise IncompatibleEmbeddingError(
                        requested=len(req.query_vector),
                        supported=self.store.dimensions,
                        message=(
                            f"Supplied query_vector dimension ({len(req.query_vector)}) does not match "
                            f"store dimension ({self.store.dimensions})."
                        ),
                    )
                query_vector = req.query_vector
            else:
                if self.embedding_service is None:
                    raise RetrievalError(
                        f"Search mode '{req.search_mode}' requires an EmbeddingService, "
                        "but none was provided at initialization."
                    )
                assert req.query is not None
                query_vector = await self.embedding_service.embed_query(req.query)

        safe_log(
            _logger,
            20,  # INFO
            "Starting retrieval request",
            event=ObservabilityEvent.RETRIEVAL_START,
            top_k=req.top_k,
            has_filters=bool(req.filters),
            has_score_threshold=req.score_threshold is not None,
            search_mode=req.search_mode,
            store_dimensions=self.store.dimensions,
            should_rerank=should_rerank,
        )

        request_deadline = time.monotonic() + self.retry_policy.timeout_seconds

        try:

            async def _do_search() -> list[RetrievalResult]:
                if req.search_mode == "dense":
                    assert query_vector is not None
                    return await self.store.search(
                        query_vector=query_vector,
                        top_k=candidate_top_k,
                        filters=req.filters,
                        score_threshold=req.score_threshold if not should_rerank else None,
                    )
                elif req.search_mode == "sparse":
                    assert req.query is not None
                    return await self.store.search_keyword(
                        query=req.query,
                        top_k=candidate_top_k,
                        filters=req.filters,
                        score_threshold=req.score_threshold if not should_rerank else None,
                    )
                else:  # hybrid
                    assert query_vector is not None
                    assert req.query is not None
                    v_task = self.store.search(
                        query_vector=query_vector,
                        top_k=candidate_top_k,
                        filters=req.filters,
                    )
                    k_task = self.store.search_keyword(
                        query=req.query,
                        top_k=candidate_top_k,
                        filters=req.filters,
                    )
                    v_res, k_res = await asyncio.gather(v_task, k_task)
                    v_weight = (
                        req.vector_weight
                        if req.vector_weight is not None
                        else settings.default_vector_weight
                    )
                    k_weight = (
                        req.keyword_weight
                        if req.keyword_weight is not None
                        else settings.default_keyword_weight
                    )
                    return weighted_score_fusion(
                        vector_results=v_res,
                        keyword_results=k_res,
                        vector_weight=v_weight,
                        keyword_weight=k_weight,
                        top_k=candidate_top_k,
                        score_threshold=req.score_threshold if not should_rerank else None,
                    )

            results = await execute_with_retry(
                _do_search,
                self.retry_policy,
                deadline=request_deadline,
            )

            reranked = False
            rerank_message: str | None = None

            if should_rerank:
                if not results:
                    pass
                elif self.reranker is None:
                    if req.on_rerank_unavailable == "raise":
                        raise RetrievalError(
                            "Re-ranking was requested, but no reranker is configured on RetrievalService."
                        )
                    safe_log(
                        _logger,
                        30,  # WARNING
                        "Re-ranking requested but no reranker configured; skipping re-ranking",
                        event=ObservabilityEvent.RETRIEVAL_RERANK_SKIPPED,
                    )
                    rerank_message = (
                        "Re-ranking requested but no reranker is configured. "
                        "Returned un-reranked retrieval candidates."
                    )
                    results = results[:final_top_k]
                else:
                    try:
                        safe_log(
                            _logger,
                            20,  # INFO
                            "Starting candidate re-ranking",
                            event=ObservabilityEvent.RETRIEVAL_RERANK_START,
                            candidate_count=len(results),
                            top_k=final_top_k,
                        )
                        rerank_query = req.query or ""
                        results = await self.reranker.rerank(
                            rerank_query, results, top_k=final_top_k
                        )
                        reranked = True
                        safe_log(
                            _logger,
                            20,  # INFO
                            "Candidate re-ranking completed",
                            event=ObservabilityEvent.RETRIEVAL_RERANK_SUCCESS,
                            final_count=len(results),
                        )
                    except MissingExtraDependencyError as err:
                        if req.on_rerank_unavailable == "raise":
                            raise
                        safe_log(
                            _logger,
                            30,  # WARNING
                            f"Re-ranking skipped due to missing dependency: {err}",
                            event=ObservabilityEvent.RETRIEVAL_RERANK_SKIPPED,
                            error=str(err),
                        )
                        rerank_message = str(err)
                        results = results[:final_top_k]
                    except Exception as err:
                        if req.on_rerank_unavailable == "raise":
                            raise
                        safe_log(
                            _logger,
                            40,  # ERROR
                            f"Re-ranking failed: {err}",
                            event=ObservabilityEvent.RETRIEVAL_RERANK_FAILED,
                            error=str(err),
                        )
                        rerank_message = (
                            f"Re-ranking failed: {err}. Returned un-reranked retrieval candidates."
                        )
                        results = results[:final_top_k]
            else:
                results = results[: req.top_k]

            # Re-apply score threshold if re-ranking produced final results
            if should_rerank and reranked and req.score_threshold is not None:
                results = [r for r in results if r.score >= req.score_threshold]
                for rank, r in enumerate(results, start=1):
                    r.rank = rank

            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

            safe_log(
                _logger,
                20,  # INFO
                "Retrieval request completed successfully",
                event=ObservabilityEvent.RETRIEVAL_SUCCESS,
                total_results=len(results),
                latency_ms=latency_ms,
                search_mode=req.search_mode,
                reranked=reranked,
            )

            return RetrievalResponse(
                results=results,
                total_results=len(results),
                latency_ms=latency_ms,
                query=req.query,
                search_mode=req.search_mode,
                reranked=reranked,
                rerank_message=rerank_message,
            )

        except (EmbeddingTimeoutError, TimeoutError) as err:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            safe_log(
                _logger,
                40,  # ERROR
                f"Retrieval request timed out: {err}",
                event=ObservabilityEvent.RETRIEVAL_FAILED,
                error=str(err),
                latency_ms=latency_ms,
            )
            raise RetrievalTimeoutError(f"Retrieval request timed out: {err}") from err

        except Exception as err:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            safe_log(
                _logger,
                40,  # ERROR
                f"Retrieval request failed: {err}",
                event=ObservabilityEvent.RETRIEVAL_FAILED,
                error=str(err),
                latency_ms=latency_ms,
            )
            raise

    async def index_documents(
        self,
        documents: list[DocumentChunk] | list[str],
        batch_size: int | None = None,
    ) -> int:
        """Convenience method to embed and index document chunks or raw texts into the store."""
        if not documents:
            return 0

        chunks_to_upsert: list[DocumentChunk] = []

        safe_log(
            _logger,
            20,  # INFO
            "Starting document indexing",
            event=ObservabilityEvent.RETRIEVAL_INDEX_START,
            total_documents=len(documents),
        )

        if isinstance(documents[0], str):
            if self.embedding_service is None:
                raise RetrievalError(
                    "Indexing raw text strings requires an EmbeddingService to generate vectors."
                )
            texts = [str(doc) for doc in documents]
            embeddings = await self.embedding_service.embed_documents(texts, batch_size=batch_size)
            for idx, (text, emb) in enumerate(zip(texts, embeddings, strict=False)):
                chunks_to_upsert.append(
                    DocumentChunk(
                        id=f"doc_{uuid.uuid4().hex[:8]}_{idx}",
                        content=text,
                        embedding=emb,
                        chunk_index=idx,
                    )
                )
        else:
            doc_chunks = [c for c in documents if isinstance(c, DocumentChunk)]
            # Find any chunks that lack embeddings
            missing_indices = [i for i, c in enumerate(doc_chunks) if c.embedding is None]
            if missing_indices:
                if self.embedding_service is None:
                    raise RetrievalError(
                        "Cannot index chunks with missing embeddings without an EmbeddingService."
                    )
                texts_to_embed = [doc_chunks[i].content for i in missing_indices]
                embeddings = await self.embedding_service.embed_documents(
                    texts_to_embed, batch_size=batch_size
                )
                for missing_idx, emb in zip(missing_indices, embeddings, strict=False):
                    doc_chunks[missing_idx].embedding = emb

            chunks_to_upsert = doc_chunks

        count = await self.store.upsert(chunks_to_upsert)

        safe_log(
            _logger,
            20,  # INFO
            "Document indexing completed successfully",
            event=ObservabilityEvent.RETRIEVAL_INDEX_SUCCESS,
            indexed_count=count,
        )

        return count

    async def delete_documents(self, chunk_ids: list[str]) -> int:
        """Delete documents from the vector store by chunk ID."""
        if not chunk_ids:
            return 0

        count = await self.store.delete(chunk_ids)

        safe_log(
            _logger,
            20,  # INFO
            "Document deletion completed successfully",
            event=ObservabilityEvent.RETRIEVAL_DELETE_SUCCESS,
            deleted_count=count,
        )

        return count
