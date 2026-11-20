"""Research worker tools and dual-storage retrieval factory."""

from __future__ import annotations

import os
from typing import Any

import httpx
from genai_core.observability import get_logger
from genai_core.retrieval.in_memory_store import InMemoryVectorStore
from genai_core.retrieval.protocol import VectorStoreAdapter
from genai_core.services.embedding_service import EmbeddingService
from genai_core.services.retrieval_service import RetrievalService
from genai_core.tools.base import FunctionTool
from genai_core.tools.registry import ToolRegistry

from ..core.config import ResearchTeamConfig
from ..exceptions import WorkerExecutionError
from .extractor import (
    EvidenceCollector,
    create_deterministic_chunk_id,
    create_deterministic_source_id,
)

_logger = get_logger("parallel_research_team.workers.tools")


def create_retrieval_service(
    config: ResearchTeamConfig,
    embedding_service: EmbeddingService | None = None,
    custom_store: VectorStoreAdapter | None = None,
) -> RetrievalService:
    """Instantiate a RetrievalService with dual-storage mode (in_memory vs postgres)."""
    store: VectorStoreAdapter

    if custom_store is not None:
        store = custom_store
    elif config.retrieval_mode == "in_memory":
        store = InMemoryVectorStore(dimensions=config.embedding_dimensions)
    elif config.retrieval_mode == "postgres":
        from genai_core.retrieval.pgvector_store import (
            PGVectorStoreAdapter,
        )

        conn_str = config.postgres_connection_string
        if not conn_str:
            raise WorkerExecutionError(
                "PostgreSQL retrieval mode requires 'postgres_connection_string' to be set."
            )

        try:
            import psycopg
        except ImportError as err:
            raise WorkerExecutionError("psycopg is required for PostgreSQL retrieval.") from err

        def _conn_factory() -> Any:
            return psycopg.AsyncConnection.connect(
                conn_str,
                autocommit=True,
            )

        store = PGVectorStoreAdapter(
            dimensions=config.embedding_dimensions,
            connection_factory=_conn_factory,
        )
    else:
        raise WorkerExecutionError(f"Unsupported retrieval mode: '{config.retrieval_mode}'.")

    return RetrievalService(
        store=store,
        embedding_service=embedding_service,
    )


def create_research_tool_registry(
    collector: EvidenceCollector,
    retrieval_service: RetrievalService | None = None,
    http_client: httpx.AsyncClient | None = None,
    tavily_api_key: str | None = None,
) -> ToolRegistry:
    """Build a ToolRegistry with research tools bound to the active EvidenceCollector."""
    registry = ToolRegistry()

    async def search_web(query: str, max_results: int = 3) -> str:
        """Search public web pages for documentation, articles, and technical updates using Tavily."""
        collector.record_query(query)
        clean_query = query.strip()
        if not clean_query:
            collector.record_error("Empty web search query submitted.")
            return "Error: Empty search query."

        api_key = tavily_api_key or os.getenv("TAVILY_API_KEY", "").strip()
        if not api_key:
            msg = "Tavily API key is not configured (TAVILY_API_KEY)."
            collector.record_error(msg)
            return f"Error: {msg}"

        try:
            results: list[dict[str, Any]] = []
            if http_client is not None:
                resp = await http_client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": api_key,
                        "query": clean_query,
                        "max_results": max_results,
                        "search_depth": "basic",
                        "include_answer": False,
                    },
                    timeout=15.0,
                )
                if resp.status_code != 200:
                    msg = f"Tavily search service returned status {resp.status_code}."
                    collector.record_error(msg)
                    return msg
                data = resp.json()
                results = data.get("results", [])
            else:
                from tavily import AsyncTavilyClient

                client = AsyncTavilyClient(api_key=api_key)
                data = await client.search(
                    query=clean_query,
                    max_results=max_results,
                    search_depth="basic",
                    include_answer=False,
                )
                results = data.get("results", [])

            if not results:
                return f"No web search results found for query: '{clean_query}'."

            formatted: list[str] = []
            for item in results:
                ref = collector.add_web_source(
                    url=item.get("url", ""),
                    title=item.get("title", item.get("url", "")),
                    content=item.get("content", ""),
                    score=float(item.get("score", 1.0)),
                )
                formatted.append(
                    f"Source ID: {ref.source_id}\n"
                    f"Chunk ID: {ref.chunk_id}\n"
                    f"Title: {ref.title}\n"
                    f"URL: {item.get('url', '')}\n"
                    f"Content: {ref.content}\n"
                )

            return "\n---\n".join(formatted)

        except Exception as exc:  # noqa: BLE001
            _logger.warning("Tavily web search error: %s", exc)
            collector.record_error(f"Tavily web search error: {exc}")
            return f"Web search could not be completed: {exc}"

    async def search_knowledge_base(query: str, top_k: int = 3) -> str:
        """Search internal documentation and pre-indexed reference manuals."""
        collector.record_query(query)
        if not retrieval_service:
            collector.record_error("Knowledge base retrieval service is not configured.")
            return "Error: Knowledge base retrieval service is not configured."

        try:
            response = await retrieval_service.search(query=query, top_k=top_k)
            if not response.results:
                return f"No relevant internal documents found for: '{query}'."

            formatted: list[str] = []
            for res in response.results:
                src_id = res.source_id or res.metadata.get(
                    "source_id",
                    create_deterministic_source_id(res.content, prefix="kb"),
                )
                chunk_id = res.id or create_deterministic_chunk_id(src_id, res.content)
                title = res.metadata.get("title", f"Internal Document ({src_id})")

                ref = collector.add_source(
                    source_id=str(src_id),
                    chunk_id=str(chunk_id),
                    title=str(title),
                    content=res.content,
                    score=res.score,
                    metadata=res.metadata,
                )
                formatted.append(
                    f"Source ID: {ref.source_id}\n"
                    f"Chunk ID: {ref.chunk_id}\n"
                    f"Title: {ref.title}\n"
                    f"Score: {ref.score:.2f}\n"
                    f"Content: {ref.content}\n"
                )

            return "\n---\n".join(formatted)

        except Exception as exc:  # noqa: BLE001
            _logger.warning("Knowledge base search error: %s", exc)
            collector.record_error(f"Knowledge base error: {exc}")
            return f"Error querying knowledge base: {exc}"

    registry.register(FunctionTool(search_web))
    registry.register(FunctionTool(search_knowledge_base))
    return registry
