"""RevisionEngine: produces improved report candidates addressing structured critique issues."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from genai_core import ObservabilityEvent, safe_log

from self_improving_report_writer.core.config import ReportWriterConfig
from self_improving_report_writer.core.exceptions import ReflectionError
from self_improving_report_writer.core.pricing import extract_or_estimate_cost
from self_improving_report_writer.models.critique import Critique
from self_improving_report_writer.models.report import Report, SourceReference
from self_improving_report_writer.prompts.manager import get_prompt_manager

if TYPE_CHECKING:
    from genai_core import LLMService

logger = logging.getLogger(__name__)

_prompt_mgr = get_prompt_manager()
REVISION_SYSTEM_PROMPT = _prompt_mgr.get_system_prompt("revision", "system")


def build_revision_prompt(
    current_report: Report,
    sources: list[SourceReference],
    critique: Critique,
) -> str:
    """Build the revision prompt containing current report, sources, and prioritized critique issues."""
    return get_prompt_manager().render_revision_prompt(
        current_report=current_report,
        sources=sources,
        critique=critique,
    )


class RevisionEngine:
    """Generates revised report candidates addressing structured critique feedback."""

    def __init__(
        self,
        llm_service: LLMService,
        config: ReportWriterConfig | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        self.llm_service = llm_service
        self.config = config or ReportWriterConfig()
        self.provider = provider or self.config.generator_provider
        self.model = model or self.config.generator_model

    async def revise_report(
        self,
        current_report: Report,
        sources: list[SourceReference],
        critique: Critique,
    ) -> tuple[Report, int, float]:
        """Revise the report addressing identified critique issues.

        Args:
            current_report: The report candidate to revise.
            sources: Verified source references available for grounding.
            critique: Structured critique with prioritized issues.

        Returns:
            Tuple of (revised_report: Report, tokens_used: int, cost_usd: float).
        """
        prompt = build_revision_prompt(
            current_report=current_report,
            sources=sources,
            critique=critique,
        )

        safe_log(
            logger,
            logging.INFO,
            f"Revising report v{current_report.version} for topic '{current_report.topic}' "
            f"(issues={len(critique.issues)}, crit={critique.critical_count}, maj={critique.major_count})",
            event=ObservabilityEvent.LLM_REQUEST_START,
            model=self.model,
        )

        try:
            response = await self.llm_service.complete_structure(
                prompt=f"{REVISION_SYSTEM_PROMPT}\n\n{prompt}",
                response_schema=Report,
                provider=self.provider,
                model=self.model,
            )
        except Exception as e:
            safe_log(
                logger,
                logging.ERROR,
                f"LLM revision call failed: {e}",
                exc_info=True,
                event=ObservabilityEvent.LLM_REQUEST_FAILED,
            )
            raise ReflectionError(f"LLM revision call raised exception: {e}") from e

        if response.error is not None:
            safe_log(
                logger,
                logging.ERROR,
                f"LLM revision returned error: {response.error.message}",
                event=ObservabilityEvent.LLM_REQUEST_FAILED,
            )
            raise ReflectionError(f"LLM revision error: {response.error.message}")

        # Extract revised report
        revised = response.structured_response
        if revised is None and response.text:
            try:
                data = json.loads(response.text)
                revised = Report.model_validate(data)
            except Exception as e:
                raise ReflectionError(f"Failed to parse text response as Report: {e}") from e

        if isinstance(revised, dict):
            try:
                revised = Report.model_validate(revised)
            except Exception as e:
                raise ReflectionError(f"Failed to validate dict response as Report: {e}") from e

        if not isinstance(revised, Report):
            raise ReflectionError(f"Expected Report instance, got {type(revised)}")

        # Ensure version is incremented and provenance recorded
        if revised.version <= current_report.version:
            revised.version = current_report.version + 1
        revised.created_by = "revision_llm"

        # Calculate tokens and cost
        _, _, tokens_used, cost_usd = extract_or_estimate_cost(
            response=response,
            model=self.model,
            default_model=self.config.generator_model,
        )

        safe_log(
            logger,
            logging.INFO,
            f"Successfully generated revised report v{revised.version} (tokens={tokens_used}, cost=${cost_usd:.4f})",
            event=ObservabilityEvent.LLM_REQUEST_SUCCESS,
        )

        return revised, tokens_used, cost_usd
