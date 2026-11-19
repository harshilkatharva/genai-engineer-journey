"""Research worker tools and dual-storage retrieval factory."""

from __future__ import annotations

from urllib.parse import unquote

import httpx
from bs4 import BeautifulSoup
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
            check_postgres_dependencies,
        )

        check_postgres_dependencies()
        if not config.postgres_connection_string:
            raise WorkerExecutionError(
                "PostgreSQL retrieval mode requires 'postgres_connection_string' to be set."
            )
        store = PGVectorStoreAdapter(
            dsn=config.postgres_connection_string,
            dimensions=config.embedding_dimensions,
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
) -> ToolRegistry:
    """Build a ToolRegistry with research tools bound to the active EvidenceCollector."""
    registry = ToolRegistry()

    async def search_web(query: str, max_results: int = 3) -> str:
        """Search public web pages for documentation, articles, and technical updates."""
        collector.record_query(query)
        if not query or not query.strip():
            collector.record_error("Empty web search query submitted.")
            return "Error: Empty search query."

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        }
        url = "https://html.duckduckgo.com/html/"
        data = {"q": query.strip()}

        try:
            if http_client is not None:
                resp = await http_client.post(url, data=data, headers=headers, timeout=10.0)
            else:
                async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                    resp = await client.post(url, data=data, headers=headers)

            if resp.status_code != 200:
                msg = f"Web search service returned status {resp.status_code}."
                collector.record_error(msg)
                return msg

            soup = BeautifulSoup(resp.text, "html.parser")
            results: list[dict[str, str]] = []

            for result_elem in soup.find_all("div", class_="result", limit=max_results * 2):
                title_elem = result_elem.find("a", class_="result__a")
                snippet_elem = result_elem.find("a", class_="result__snippet")

                if title_elem and snippet_elem:
                    title = title_elem.get_text(strip=True)
                    href_val = title_elem.get("href", "")
                    raw_href = href_val if isinstance(href_val, str) else str(href_val or "")
                    snippet = snippet_elem.get_text(strip=True)

                    clean_url = raw_href
                    if "uddg=" in raw_href:
                        part = raw_href.split("uddg=")[-1].split("&")[0]
                        clean_url = unquote(part)

                    results.append({"title": title, "url": clean_url, "snippet": snippet})
                    if len(results) >= max_results:
                        break

            if not results:
                return f"No web search results found for query: '{query}'."

            formatted: list[str] = []
            for item in results:
                ref = collector.add_web_source(
                    url=item["url"],
                    title=item["title"],
                    content=item["snippet"],
                )
                formatted.append(
                    f"Source ID: {ref.source_id}\n"
                    f"Chunk ID: {ref.chunk_id}\n"
                    f"Title: {ref.title}\n"
                    f"URL: {item['url']}\n"
                    f"Content: {ref.content}\n"
                )

            return "\n---\n".join(formatted)

        except Exception as exc:  # noqa: BLE001
            _logger.warning("Web search error: %s", exc)
            collector.record_error(f"Web search error: {exc}")
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
