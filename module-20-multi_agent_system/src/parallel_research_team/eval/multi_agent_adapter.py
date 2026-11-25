"""Adapter wrapping Module 20 ParallelResearchTeam for multi-agent evaluation."""

from __future__ import annotations

import time

from genai_core import LLMService, RetrievalService
from genai_core.observability import get_logger

from ..core.config import ResearchTeamConfig, get_research_team_config
from ..models import Report
from ..orchestrator import ParallelResearchTeam
from ..storage.file_store import FileCheckpointStore
from .models import AgentExecutionMetrics
from .rate_limiter import AsyncRateLimiter, get_global_rate_limiter

_logger = get_logger("parallel_research_team.eval.multi_agent")


def format_report_to_text(report: Report | None) -> str:
    """Format structured Report into cohesive markdown text for evaluation."""
    if not report:
        return ""

    blocks: list[str] = [f"# {report.title}\n"]

    if report.executive_summary and report.executive_summary.content:
        blocks.append(f"## Executive Summary\n{report.executive_summary.content}\n")

    for section in report.sections:
        blocks.append(f"## {section.title}\n{section.content}\n")
        if section.claims:
            claims_text = []
            for c in section.claims:
                cites = [f"[{cit.source_id}]" for cit in c.citations]
                claims_text.append(f"- {c.text} {' '.join(cites)}")
            blocks.append("### Claims & Grounding\n" + "\n".join(claims_text) + "\n")

    if report.conclusion and report.conclusion.content:
        blocks.append(f"## Conclusion\n{report.conclusion.content}\n")

    if report.limitations:
        lims = "\n".join(f"- {lim}" for lim in report.limitations)
        blocks.append(f"## Limitations\n{lims}\n")

    return "\n".join(blocks).strip()


class MultiAgentAdapter:
    """Executes multi-agent research tasks using Module 20's ParallelResearchTeam."""

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

        self.checkpoint_store = FileCheckpointStore(config=self.config)
        self.team = ParallelResearchTeam(
            config=self.config,
            llm_service=self.llm_service,
            retrieval_service=self.retrieval_service,
            checkpoint_store=self.checkpoint_store,
        )

    async def run(self, query: str) -> tuple[str, AgentExecutionMetrics]:
        """Execute query using ParallelResearchTeam and return report text with metrics."""
        await self.rate_limiter.acquire()
        start_time = time.perf_counter()

        try:
            result = await self.team.run(topic=query)
            elapsed_s = round(time.perf_counter() - start_time, 2)

            final_text = format_report_to_text(result.report)
            if not final_text and result.error_message:
                final_text = f"Execution error: {result.error_message}"

            # Calculate total worker tool steps & sources
            tool_calls = 0
            sources_count = 0
            if result.metrics and result.metrics.worker_breakdown:
                for w_info in result.metrics.worker_breakdown.values():
                    tool_calls += int(w_info.get("sources_count", 0)) + 1
                    sources_count += int(w_info.get("sources_count", 0))

            checkpoint = await self.checkpoint_store.get_checkpoint(result.run_id)
            if checkpoint and checkpoint.authoritative_sources:
                sources_count = len(checkpoint.authoritative_sources)

            metrics = AgentExecutionMetrics(
                system_name="multi_agent",
                status=result.status.value,
                tokens=result.metrics.total_tokens,
                cost_usd=round(result.metrics.total_cost_usd, 6),
                latency_seconds=elapsed_s,
                tool_calls_count=tool_calls,
                sources_count=sources_count,
                error_message=result.error_message,
            )
            return final_text, metrics
        except Exception as exc:
            elapsed_s = round(time.perf_counter() - start_time, 2)
            _logger.exception("MultiAgent execution failed for query '%s'", query)
            metrics = AgentExecutionMetrics(
                system_name="multi_agent",
                status="failed",
                tokens=0,
                cost_usd=0.0,
                latency_seconds=elapsed_s,
                tool_calls_count=0,
                sources_count=0,
                error_message=str(exc),
            )
            return f"Execution error: {exc}", metrics
