"""Lead Agent synthesis engine compiling verified worker evidence into structured reports."""

from __future__ import annotations

import asyncio

from genai_core import LLMService
from genai_core.observability import get_logger

from ..core.budget import SharedRunBudget
from ..core.config import ResearchTeamConfig, get_research_team_config
from ..exceptions import SynthesisError
from ..models import CitationType, Report, WorkerExecutionResult, WorkerStatus
from ..prompts.manager import PromptTemplateManager, get_prompt_manager
from ..telemetry.pricing import PricingCalculator
from .source_pool import AuthoritativeSourcePool

_logger = get_logger("parallel_research_team.synthesis.engine")


class SynthesisEngine:
    """Lead Agent component that synthesizes multi-worker findings into a structured research Report."""

    def __init__(
        self,
        config: ResearchTeamConfig | None = None,
        llm_service: LLMService | None = None,
        prompt_manager: PromptTemplateManager | None = None,
        pricing_calculator: PricingCalculator | None = None,
    ) -> None:
        self.config = config or get_research_team_config()
        self.llm_service = llm_service or LLMService(
            default_provider=self.config.synthesis_provider,
            default_model=self.config.synthesis_model,
        )
        self.prompt_manager = prompt_manager or get_prompt_manager(self.config)
        self.pricing_calculator = pricing_calculator or PricingCalculator()
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0
        self.last_total_tokens: int = 0
        self.last_cost_usd: float = 0.0

    async def synthesize(
        self,
        topic: str,
        worker_results: dict[str, WorkerExecutionResult],
        source_pool: AuthoritativeSourcePool,
        shared_budget: SharedRunBudget | None = None,
    ) -> Report:
        """Synthesize collective findings and sources into a structured Report."""
        clean_topic = topic.strip()
        if not clean_topic:
            raise SynthesisError("Cannot synthesize report: topic cannot be empty.")

        # Check shared budget before dispatching expensive synthesis LLM call
        if shared_budget is not None:
            await shared_budget.check_budget()

        # 1. Process worker deliverables, extract findings, and document worker failures
        worker_findings: dict[str, str] = {}
        unresolved_limitations: list[str] = []
        successful_count = 0

        for brief_id, res in worker_results.items():
            if res.status == WorkerStatus.COMPLETED and res.output:
                worker_findings[brief_id] = res.output.findings_summary
                successful_count += 1
            elif res.status == WorkerStatus.PARTIAL and res.output:
                worker_findings[brief_id] = f"[PARTIAL EVIDENCE] {res.output.findings_summary}"
                err_note = f"Brief '{brief_id}' partially completed: {res.error_message or 'Step limit reached'}"
                unresolved_limitations.append(err_note)
                successful_count += 1
            else:
                err_msg = res.error_message or f"Worker encountered status: {res.status.value}"
                worker_findings[brief_id] = f"[RESEARCH FAILED / TIMED OUT] Reason: {err_msg}"
                fail_note = f"Brief '{brief_id}' research incomplete: {err_msg}"
                unresolved_limitations.append(fail_note)

        if successful_count == 0 and len(source_pool) == 0:
            raise SynthesisError(
                "Cannot synthesize report: all parallel workers failed and zero authoritative sources were collected."
            )

        # 2. Render versioned prompt templates
        system_prompt, user_prompt = self.prompt_manager.render_synthesis_prompts(
            topic=clean_topic,
            worker_findings=worker_findings,
            sources=source_pool.list_sources(),
            version=self.config.prompt_version,
        )

        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        _logger.info(
            "Invoking Lead Agent synthesis for topic '%s' using %s/%s with %d pooled sources",
            clean_topic,
            self.config.synthesis_provider,
            self.config.synthesis_model,
            len(source_pool),
        )

        # 3. Request structured completion conforming to Report schema with transient retry
        max_attempts = 3
        response = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = await self.llm_service.complete_structure(
                    prompt=full_prompt,
                    response_schema=Report,
                    provider=self.config.synthesis_provider,
                    model=self.config.synthesis_model,
                )
            except Exception as exc:
                if attempt == max_attempts:
                    _logger.exception("LLM structured synthesis failed")
                    raise SynthesisError(f"LLM synthesis call failed: {exc}") from exc
                await asyncio.sleep(2.0 * attempt)
                continue

            err = getattr(response, "error", None)
            if err is not None and isinstance(getattr(err, "message", None), str):
                err_msg = err.message.lower()
                is_transient = any(
                    k in err_msg
                    for k in [
                        "high demand",
                        "rate limit",
                        "503",
                        "429",
                        "resource exhausted",
                        "try again",
                    ]
                )
                if is_transient and attempt < max_attempts:
                    _logger.warning(
                        "Transient LLM error on synthesis attempt %d/%d: %s. Retrying in %.1fs...",
                        attempt,
                        max_attempts,
                        err.message,
                        3.0 * attempt,
                    )
                    await asyncio.sleep(3.0 * attempt)
                    continue
                raise SynthesisError(f"LLM synthesis error: {err.message}")
            break

        if response is None:
            raise SynthesisError("LLM synthesis failed to produce a response.")

        report = response.structured_response
        if not isinstance(report, Report):
            raise SynthesisError(
                f"LLM response schema mismatch: expected Report, got {type(report).__name__}."
            )

        # 4. Record token usage and expenditure
        in_tokens = response.usage.get("input_tokens", 0) if response.usage else 0
        out_tokens = response.usage.get("output_tokens", 0) if response.usage else 0
        total_tokens = in_tokens + out_tokens
        cost_usd = self.pricing_calculator.calculate_cost(
            self.config.synthesis_model, in_tokens, out_tokens
        )
        self.last_input_tokens = in_tokens
        self.last_output_tokens = out_tokens
        self.last_total_tokens = total_tokens
        self.last_cost_usd = cost_usd

        if shared_budget is not None and total_tokens > 0:
            await shared_budget.record_usage(tokens=total_tokens, cost_usd=cost_usd)

        # 5. Populate and reconcile Module 20 metadata and unresolved worker limitations
        for limitation in unresolved_limitations:
            if limitation not in report.unresolved_limitations:
                report.unresolved_limitations.append(limitation)
            if limitation not in report.limitations:
                report.limitations.append(limitation)

        # Populate brief coverages if missing
        if not report.brief_coverages:
            from ..models.report import BriefCoverage

            report.brief_coverages = [
                BriefCoverage(
                    brief_id=b_id,
                    coverage_summary=f"Synthesized coverage for brief {b_id} (status: {worker_results[b_id].status.value})",
                )
                for b_id in worker_results
            ]

        # 6. Post-Synthesis Claim Alignment & Citation Reconciler
        valid_source_chunks: dict[tuple[str, str], str] = {
            (src.source_id, src.chunk_id): src.content for src in source_pool.list_sources()
        }

        all_sections = [report.executive_summary] + report.sections + [report.conclusion]
        for section in all_sections:
            if not section:
                continue
            for claim in section.claims:
                claim.section_id = section.section_id
                valid_citations = []
                for cite in claim.citations:
                    chunk_key = (cite.source_id, cite.chunk_id)
                    if chunk_key in valid_source_chunks:
                        if (
                            cite.citation_type
                            and cite.citation_type.value == "quotation"
                            and cite.quoted_passage
                        ):
                            chunk_text = valid_source_chunks[chunk_key]
                            if cite.quoted_passage not in chunk_text:
                                cite.citation_type = CitationType.PARAPHRASE
                                cite.quoted_passage = None
                        valid_citations.append(cite)
                    else:
                        _logger.warning(
                            "Pruning invalid citation %s referencing missing chunk (%s, %s)",
                            cite.citation_id,
                            cite.source_id,
                            cite.chunk_id,
                        )
                claim.citations = valid_citations
                if claim.is_factual and not claim.citations:
                    claim.is_factual = False

        report.research_team_version = self.config.system_version
        return report
