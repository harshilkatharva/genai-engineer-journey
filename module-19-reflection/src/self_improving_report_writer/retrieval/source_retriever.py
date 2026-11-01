"""Evidence retrieval adapter integrating with genai-core RetrievalService."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from genai_core import ObservabilityEvent, safe_log

from ..core.exceptions import RetrievalError
from ..models.report import SourceReference

if TYPE_CHECKING:
    from genai_core import DocumentChunk, RetrievalService, VectorStoreAdapter

_logger = logging.getLogger("self_improving_report_writer.retrieval")


class SourceRetriever:
    """Facade over genai-core.RetrievalService for evidence gathering and chunk management."""

    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        vector_store: VectorStoreAdapter | None = None,
    ) -> None:
        """Initialize retriever with an existing service or create an in-memory fallback."""
        if retrieval_service is not None:
            self.service = retrieval_service
        else:
            from genai_core import InMemoryVectorStore
            from genai_core import RetrievalService as CoreRetrievalService

            store = vector_store or InMemoryVectorStore(dimensions=768)
            self.service = CoreRetrievalService(store=store)

    async def retrieve_sources(
        self,
        topic: str,
        top_k: int = 8,
        min_score: float = 0.0,
    ) -> list[SourceReference]:
        """Retrieve relevant source chunks for a topic from the vector store.

        Args:
            topic: Natural language topic or research query.
            top_k: Maximum number of chunks to return.
            min_score: Minimum relevance score threshold (0.0 to 1.0).

        Returns:
            List of immutable SourceReference objects.
        """
        trimmed_topic = topic.strip()
        if not trimmed_topic:
            raise RetrievalError("Topic cannot be empty or pure whitespace")

        safe_log(
            _logger,
            logging.INFO,
            f"Retrieving sources for topic: '{trimmed_topic[:60]}...'",
            event=ObservabilityEvent.RETRIEVAL_START,
            topic=trimmed_topic,
            top_k=top_k,
        )

        try:
            response = await self.service.search(query=trimmed_topic, top_k=top_k)
        except Exception as exc:
            safe_log(
                _logger,
                logging.ERROR,
                f"Retrieval query failed: {exc}",
                event=ObservabilityEvent.RETRIEVAL_FAILED,
                error=str(exc),
            )
            raise RetrievalError(f"Evidence retrieval failed: {exc}") from exc

        sources: list[SourceReference] = []
        for res in response.results:
            if res.score < min_score:
                continue

            meta = dict(res.metadata) if res.metadata else {}
            src_id = res.source_id or meta.get("source_id") or f"src_{res.id}"
            title = meta.get("title") or f"Document {src_id}"

            sources.append(
                SourceReference(
                    source_id=str(src_id),
                    chunk_id=str(res.id),
                    title=str(title),
                    content=res.content,
                    score=float(res.score),
                    metadata=meta,
                )
            )

        safe_log(
            _logger,
            logging.INFO,
            f"Retrieved {len(sources)} source chunks for topic.",
            event=ObservabilityEvent.RETRIEVAL_SUCCESS,
            count=len(sources),
        )

        return sources

    async def index_sources(
        self,
        documents: list[DocumentChunk] | list[dict[str, Any]],
    ) -> int:
        """Index raw documents or chunks into the vector store.

        Convenience utility for testing and topic ingestion.
        """
        from genai_core import DocumentChunk

        chunks: list[DocumentChunk] = []
        for doc in documents:
            if isinstance(doc, DocumentChunk):
                chunks.append(doc)
            elif isinstance(doc, dict):
                chunks.append(
                    DocumentChunk(
                        id=str(doc.get("id", doc.get("chunk_id", ""))),
                        content=str(doc.get("content", "")),
                        source_id=doc.get("source_id"),
                        metadata=doc.get("metadata", {}),
                    )
                )

        return await self.service.index_documents(chunks)
