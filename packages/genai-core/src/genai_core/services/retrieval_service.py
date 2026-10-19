from __future__ import annotations

import time
import uuid
from typing import Any

from ..core import get_settings
from ..exceptions import (
    EmbeddingTimeoutError,
    IncompatibleEmbeddingError,
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
from ..retrieval.protocol import VectorStoreAdapter
from .embedding_service import EmbeddingService

_logger = get_logger("genai_core.services.retrieval_service")


class RetrievalService:
    """Orchestrates query validation, embedding, resilient vector search, and ranking."""

    def __init__(
        self,
        store: VectorStoreAdapter,
        embedding_service: EmbeddingService | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.store = store
        self.embedding_service = embedding_service

        settings = get_settings()

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
        """Execute semantic search against the configured vector store.

        Accepts either a structured RetrievalQuery or a string query with keyword overrides.
        """
        start_time = time.perf_counter()

        if isinstance(query, str):
            req = RetrievalQuery(query=query, **kwargs)
        elif isinstance(query, RetrievalQuery):
            req = query
        else:
            raise TypeError(
                f"Expected query to be a str or RetrievalQuery, got {type(query).__name__}."
            )

        # Enforce supported search mode (dense search only in Phase 3)
        if req.search_mode != "dense":
            raise UnsupportedSearchModeError(req.search_mode, supported_modes=["dense"])

        # Determine query vector
        query_vector: list[float]
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
                    "Text query requires an EmbeddingService, "
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
        )

        request_deadline = time.monotonic() + self.retry_policy.timeout_seconds

        try:

            async def _do_search() -> list[RetrievalResult]:
                return await self.store.search(
                    query_vector=query_vector,
                    top_k=req.top_k,
                    filters=req.filters,
                    score_threshold=req.score_threshold,
                )

            results = await execute_with_retry(
                _do_search,
                self.retry_policy,
                deadline=request_deadline,
            )

            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

            safe_log(
                _logger,
                20,  # INFO
                "Retrieval request completed successfully",
                event=ObservabilityEvent.RETRIEVAL_SUCCESS,
                total_results=len(results),
                latency_ms=latency_ms,
                search_mode=req.search_mode,
            )

            return RetrievalResponse(
                results=results,
                total_results=len(results),
                latency_ms=latency_ms,
                query=req.query,
                search_mode=req.search_mode,
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
