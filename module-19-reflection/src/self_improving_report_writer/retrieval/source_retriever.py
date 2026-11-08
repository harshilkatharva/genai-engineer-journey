"""Evidence retrieval adapter integrating with genai-core RetrievalService."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from genai_core import ObservabilityEvent, safe_log

from ..core.config import ReportWriterConfig, get_default_config
from ..core.exceptions import RetrievalError
from ..models.report import SourceReference

if TYPE_CHECKING:
    from genai_core import DocumentChunk, EmbeddingService, RetrievalService, VectorStoreAdapter

_logger = logging.getLogger("self_improving_report_writer.retrieval")


class SourceRetriever:
    """Facade over genai-core.RetrievalService for evidence gathering and chunk management."""

    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        vector_store: VectorStoreAdapter | None = None,
        embedding_service: EmbeddingService | None = None,
        config: ReportWriterConfig | None = None,
    ) -> None:
        """Initialize retriever with PostgreSQL runtime defaults or in-memory testing store."""
        self.config = config or get_default_config()

        if retrieval_service is not None:
            self.service = retrieval_service
            self.store = getattr(retrieval_service, "store", vector_store)
            self.embedding_service = getattr(
                retrieval_service, "embedding_service", embedding_service
            )
        else:
            from genai_core import RetrievalService as CoreRetrievalService

            # Configure embedding service
            if embedding_service is not None:
                self.embedding_service = embedding_service
            else:
                from genai_core import EmbeddingService as CoreEmbeddingService

                try:
                    self.embedding_service = CoreEmbeddingService(
                        provider=self.config.embedding_provider,
                        model=self.config.embedding_model,
                        dimensions=self.config.embedding_dimensions,
                    )
                except Exception as exc:
                    safe_log(
                        _logger,
                        logging.ERROR,
                        f"Failed to initialize required EmbeddingService: {exc}",
                        event=ObservabilityEvent.RETRIEVAL_FAILED,
                        error=str(exc),
                    )
                    raise RetrievalError(
                        f"Failed to initialize required EmbeddingService: {exc}"
                    ) from exc

            # Configure vector store adapter
            if vector_store is not None:
                self.store = vector_store
            elif self.config.vector_store_backend == "pgvector":
                import psycopg
                from genai_core import PGVectorStoreAdapter

                def _conn_factory() -> Any:
                    return psycopg.AsyncConnection.connect(self.config.database_url)

                self.store = PGVectorStoreAdapter(
                    dimensions=self.config.embedding_dimensions,
                    table_name=self.config.vector_store_table,
                    connection_factory=_conn_factory,
                )
            elif self.config.vector_store_backend == "memory":
                from genai_core import InMemoryVectorStore

                self.store = InMemoryVectorStore(dimensions=self.config.embedding_dimensions)
            else:
                raise ValueError(
                    f"Unsupported vector_store_backend: '{self.config.vector_store_backend}'. "
                    "Expected 'pgvector' or 'memory'."
                )

            self.service = CoreRetrievalService(
                store=self.store,
                embedding_service=self.embedding_service,
            )

        self.retrieval_service = self.service

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

        try:
            return await self.service.index_documents(chunks)
        except Exception as exc:
            safe_log(
                _logger,
                logging.ERROR,
                f"Evidence indexing failed: {exc}",
                event=ObservabilityEvent.RETRIEVAL_FAILED,
                error=str(exc),
            )
            raise RetrievalError(f"Evidence indexing failed: {exc}") from exc
