"""Bounded default research worker implementation."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any

import httpx
from genai_core import (
    ChatMessage,
    LLMService,
    ToolBudget,
    ToolExecutor,
    ToolLoopResult,
    ToolLoopRunner,
    ToolLoopStatus,
)
from genai_core.observability import get_logger
from genai_core.services.retrieval_service import RetrievalService

from ..core.budget import SharedRunBudget
from ..core.config import ResearchTeamConfig, get_research_team_config
from ..models import (
    WorkerBrief,
    WorkerExecutionResult,
    WorkerResearchOutput,
    WorkerStatus,
)
from ..prompts.manager import PromptTemplateManager
from ..telemetry.pricing import PricingCalculator
from .extractor import EvidenceCollector, EvidenceExtractor
from .protocol import ResearchWorkerProtocol
from .tools import create_research_tool_registry

_logger = get_logger("parallel_research_team.workers.worker")


class DefaultResearchWorker(ResearchWorkerProtocol):
    """Specialized bounded research agent wrapping ToolLoopRunner.

    Enforces step boundaries, per-worker timeouts, shared run budget deductions,
    and isolates failures by returning typed WorkerExecutionResult.
    """

    def __init__(
        self,
        config: ResearchTeamConfig | None = None,
        llm_service: LLMService | None = None,
        retrieval_service: RetrievalService | None = None,
        prompt_manager: PromptTemplateManager | None = None,
        pricing_calculator: PricingCalculator | None = None,
        evidence_extractor: EvidenceExtractor | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config or get_research_team_config()
        self.llm_service = llm_service or LLMService(
            default_provider=self.config.default_provider,
            default_model=self.config.default_model,
        )
        self.retrieval_service = retrieval_service
        self.prompt_manager = prompt_manager or PromptTemplateManager(config=self.config)
        self.pricing_calculator = pricing_calculator or PricingCalculator()
        self.evidence_extractor = evidence_extractor or EvidenceExtractor()
        self.http_client = http_client

    @property
    def worker_type(self) -> str:
        """Worker type identifier."""
        return "default"

    async def execute(
        self,
        brief: WorkerBrief,
        shared_budget: SharedRunBudget,
    ) -> WorkerExecutionResult:
        """Execute assigned research brief bounded by steps, timeout, and shared budget."""
        started_at = datetime.now(UTC)
        start_mono = time.monotonic()
        effective_timeout = min(brief.timeout_seconds, self.config.per_worker_timeout_seconds)

        # 1. Proactive budget check
        if not await shared_budget.can_proceed():
            return WorkerExecutionResult(
                brief_id=brief.brief_id,
                status=WorkerStatus.FAILED,
                error_message="Shared run budget already exhausted before worker execution.",
                started_at=started_at,
                completed_at=datetime.now(UTC),
                duration_ms=round((time.monotonic() - start_mono) * 1000.0, 2),
            )

        # 2. Initialize per-worker evidence collector and tools
        collector = EvidenceCollector()
        tool_registry = create_research_tool_registry(
            collector=collector,
            retrieval_service=self.retrieval_service,
            http_client=self.http_client,
            tavily_api_key=self.config.tavily_api_key,
        )

        executor = ToolExecutor(
            registry=tool_registry,
            default_timeout_s=effective_timeout,
            max_concurrency=2,
        )

        runner = ToolLoopRunner(
            llm_service=self.llm_service,
            executor=executor,
            max_iterations=brief.max_steps,
            default_timeout_s=effective_timeout,
        )

        # 3. Render prompt templates
        system_prompt, user_prompt = self.prompt_manager.render_worker_prompts(
            brief=brief,
            version=self.config.prompt_version,
        )
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        worker_budget = ToolBudget(
            max_steps=brief.max_steps,
            deadline_monotonic=time.monotonic() + effective_timeout,
        )

        loop_result: ToolLoopResult | None = None
        status: WorkerStatus = WorkerStatus.FAILED
        error_message: str | None = None

        # 4. Execute tool loop with wall-clock timeout protection
        try:
            loop_result = await asyncio.wait_for(
                runner.run(
                    messages=messages,
                    budget=worker_budget,
                    provider=self.config.default_provider,
                    model=self.config.default_model,
                ),
                timeout=effective_timeout,
            )
        except TimeoutError:
            status = WorkerStatus.TIMEOUT
            error_message = f"Worker timed out after {effective_timeout:.1f}s."
            _logger.warning("Worker for brief '%s' timed out", brief.brief_id)
        except Exception as exc:  # noqa: BLE001
            status = WorkerStatus.FAILED
            error_message = f"Worker execution error: {exc}"
            _logger.error("Worker for brief '%s' failed with exception: %s", brief.brief_id, exc)

        duration_ms = round((time.monotonic() - start_mono) * 1000.0, 2)
        completed_at = datetime.now(UTC)

        # 5. Calculate token usage and cost using exact counts
        input_tokens = (
            loop_result.input_tokens if loop_result else worker_budget.current_input_tokens
        )
        output_tokens = (
            loop_result.output_tokens if loop_result else worker_budget.current_output_tokens
        )
        total_tokens = loop_result.total_tokens if loop_result else worker_budget.current_tokens

        cost_usd = self.pricing_calculator.calculate_cost(
            self.config.default_model, input_tokens, output_tokens
        )

        # 6. Deduct from shared budget atomically
        await shared_budget.record_usage(tokens=total_tokens, cost_usd=cost_usd)

        # 7. Map status and extract output (with partial evidence salvage)
        output: WorkerResearchOutput | None = None
        if loop_result is not None:
            if loop_result.status == ToolLoopStatus.COMPLETED:
                status = WorkerStatus.COMPLETED
            elif loop_result.status == ToolLoopStatus.MAX_ITERATIONS_REACHED:
                status = WorkerStatus.PARTIAL
                error_message = f"Step limit ({brief.max_steps}) reached."
            elif loop_result.status in (
                ToolLoopStatus.BUDGET_EXCEEDED,
                ToolLoopStatus.DEADLINE_EXCEEDED,
            ):
                status = WorkerStatus.PARTIAL if collector.has_sources() else WorkerStatus.TIMEOUT
                error_message = (
                    loop_result.error_message or "Execution budget or deadline exceeded."
                )
            elif loop_result.status == ToolLoopStatus.FAILED:
                status = WorkerStatus.FAILED
                error_message = loop_result.error_message or "Tool loop failed."

            output = self.evidence_extractor.extract(
                brief=brief,
                final_text=loop_result.final_text,
                collector=collector,
                additional_limitations=[error_message] if error_message else None,
            )
        elif collector.has_sources():
            # Salvage accumulated evidence on timeout or exception
            output = self.evidence_extractor.extract(
                brief=brief,
                final_text=f"Execution interrupted ({status.value}): {error_message}",
                collector=collector,
                additional_limitations=[error_message] if error_message else None,
            )

        return WorkerExecutionResult(
            brief_id=brief.brief_id,
            status=status,
            output=output,
            tokens_used=total_tokens,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
            error_message=error_message,
            started_at=started_at,
            completed_at=completed_at,
        )


def default_worker_factory(
    brief: WorkerBrief | None = None,
    config: ResearchTeamConfig | None = None,
    **kwargs: Any,
) -> ResearchWorkerProtocol:
    """Factory function for creating DefaultResearchWorker instances.

    Note:
        DefaultResearchWorker is stateless with respect to the brief during initialization;
        the brief is supplied dynamically to worker.execute(brief, budget).
        The brief parameter is accepted to adhere to the extensible WorkerFactory protocol.
    """
    _ = brief  # Handled dynamically in worker.execute()
    return DefaultResearchWorker(config=config, **kwargs)
