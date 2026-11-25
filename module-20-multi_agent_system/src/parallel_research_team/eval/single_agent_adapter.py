"""Adapter wrapping Module 17 BoundedAgent for fair single-agent evaluation."""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from genai_core import (
    InMemoryMemoryStore,
    LLMService,
    MemoryService,
    RetrievalService,
    ToolBudget,
    ToolRegistry,
)
from genai_core.observability import get_logger
from genai_core.tools.base import FunctionTool
from research_and_support_agent.agent import BoundedAgent
from research_and_support_agent.core.config import AgentConfig
from research_and_support_agent.memory.manager import AgentMemoryManager

from ..core.config import ResearchTeamConfig, get_research_team_config
from .models import AgentExecutionMetrics
from .rate_limiter import AsyncRateLimiter, get_global_rate_limiter

load_dotenv()
_logger = get_logger("parallel_research_team.eval.single_agent")


def create_single_agent_tool_registry(
    retrieval_service: RetrievalService | None = None,
    tavily_api_key: str | None = None,
) -> ToolRegistry:
    """Construct ToolRegistry equipped with Tavily web search and Knowledge Base retrieval."""
    registry = ToolRegistry()

    # 1. Register Tavily search_web
    async def search_web(query: str, max_results: int = 3) -> str:
        """Search public web pages for technical documentation, external guidelines, and updates."""
        clean_q = query.strip()
        if not clean_q:
            return "Error: Empty search query."

        api_key = tavily_api_key or os.getenv("TAVILY_API_KEY", "").strip()
        if not api_key:
            return "Error: Tavily API key is not configured (TAVILY_API_KEY)."

        try:
            from tavily import AsyncTavilyClient

            client = AsyncTavilyClient(api_key=api_key)
            data = await client.search(
                query=clean_q,
                max_results=max_results,
                search_depth="basic",
                include_answer=False,
            )
            results = data.get("results", [])
            if not results:
                return f"No web search results found for query: '{clean_q}'."

            formatted: list[str] = []
            for idx, item in enumerate(results, 1):
                title = item.get("title", "")
                url = item.get("url", "")
                snippet = item.get("content") or item.get("snippet") or title
                formatted.append(f"[{idx}] {title}\nURL: {url}\n{snippet}")

            return "\n---\n".join(formatted)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Single agent Tavily search error: %s", exc)
            return f"Web search could not be completed: {exc}"

    registry.register(
        FunctionTool(
            func=search_web,
            name="search_web",
            description="Search public web pages for technical documentation, external guidelines, updates and any web pages content.",
            tags=["research", "web"],
        )
    )

    # 2. Register search_knowledge_base
    async def search_knowledge_base(query: str, top_k: int = 3) -> str:
        """Search internal documentation and policy manuals for company guidelines and technical rules."""
        if not retrieval_service:
            return "Error: Knowledge base retrieval service is not configured."
        try:
            resp = await retrieval_service.search(query=query, top_k=top_k)
            if not resp or not resp.chunks:
                return f"No knowledge base documents found for query: '{query}'."
            formatted = [f"[{c.chunk_id}] {c.content}" for c in resp.chunks]
            return "\n---\n".join(formatted)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Single agent KB retrieval error: %s", exc)
            return f"Knowledge base search failed: {exc}"

    registry.register(
        FunctionTool(
            func=search_knowledge_base,
            name="search_knowledge_base",
            description="Search internal documentation and handbook chunks for technical rules and concepts.",
            tags=["research", "kb"],
        )
    )

    return registry


class SingleAgentAdapter:
    """Executes single-agent research tasks using Module 17's BoundedAgent."""

    def __init__(
        self,
        config: ResearchTeamConfig | None = None,
        retrieval_service: RetrievalService | None = None,
        llm_service: LLMService | None = None,
        rate_limiter: AsyncRateLimiter | None = None,
    ) -> None:
        self.config = config or get_research_team_config()
        self.retrieval_service = retrieval_service
        self.rate_limiter = rate_limiter or get_global_rate_limiter()

        self.llm_service = llm_service or LLMService(
            default_provider=self.config.default_provider,
            default_model=self.config.default_model,
        )

        # In-memory session store & memory manager
        self.memory_service = MemoryService(store=InMemoryMemoryStore())
        self.memory_manager = AgentMemoryManager(memory_service=self.memory_service)

        # Construct shared tool registry using Tavily & Knowledge base
        self.tool_registry = create_single_agent_tool_registry(
            retrieval_service=self.retrieval_service,
            tavily_api_key=self.config.tavily_api_key or os.getenv("TAVILY_API_KEY"),
        )

        self.agent_config = AgentConfig(
            default_max_steps=8,
            default_max_tokens=30_000,
            default_max_cost_usd=0.05,
            default_timeout_seconds=60.0,
            default_max_repeated_calls=3,
        )

        self.agent = BoundedAgent(
            llm_service=self.llm_service,
            memory_manager=self.memory_manager,
            retrieval_service=self.retrieval_service,
            tool_registry=self.tool_registry,
            config=self.agent_config,
        )

    async def run(self, query: str) -> tuple[str, AgentExecutionMetrics]:
        """Execute query using single bounded agent and return answer text with metrics."""
        await self.rate_limiter.acquire()
        start_time = time.perf_counter()

        system_prompt = (
            "You are a professional technical research assistant. "
            "Investigate the user's research question thoroughly using search_web and search_knowledge_base. "
            "Synthesize a clear, structured, well-grounded answer based directly on the retrieved evidence. "
            "Be precise, factual, and avoid unsubstantiated claims."
        )

        budget = ToolBudget(
            max_steps=self.agent_config.default_max_steps,
            max_tokens=self.agent_config.default_max_tokens,
            max_cost_usd=self.agent_config.default_max_cost_usd,
            deadline_monotonic=time.monotonic() + self.agent_config.default_timeout_seconds,
        )

        try:
            result = await self.agent.run(
                query=query,
                system_prompt=system_prompt,
                budget=budget,
            )
            elapsed_s = round(time.perf_counter() - start_time, 2)
            final_text = (result.final_text or "").strip()
            if not final_text and result.messages:
                # Fallback to last assistant message if final_text is empty
                for m in reversed(result.messages):
                    if m.role == "assistant" and m.content:
                        final_text = m.content.strip()
                        break

            # Count tool invocations
            tool_calls = sum(
                len(m.tool_calls or []) for m in result.messages if m.role == "assistant"
            )

            metrics = AgentExecutionMetrics(
                system_name="single_agent",
                status=str(
                    result.status.value if hasattr(result.status, "value") else result.status
                ),
                tokens=result.total_tokens,
                cost_usd=round(result.total_cost_usd, 6),
                latency_seconds=elapsed_s,
                tool_calls_count=tool_calls,
                sources_count=tool_calls,  # Approximate evidence steps
                error_message=result.error_message,
            )
            return final_text, metrics
        except Exception as exc:
            elapsed_s = round(time.perf_counter() - start_time, 2)
            _logger.exception("SingleAgent execution failed for query '%s'", query)
            metrics = AgentExecutionMetrics(
                system_name="single_agent",
                status="failed",
                tokens=0,
                cost_usd=0.0,
                latency_seconds=elapsed_s,
                tool_calls_count=0,
                sources_count=0,
                error_message=str(exc),
            )
            return f"Execution error: {exc}", metrics
