"""Report generator synthesizing structured reports via LLMService."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from genai_core import ObservabilityEvent, safe_log

from ..core.config import ReportWriterConfig, get_default_config
from ..core.exceptions import GenerationError
from ..models.report import Report, SourceReference
from .prompts import build_report_generation_prompt

if TYPE_CHECKING:
    from genai_core import LLMService

_logger = logging.getLogger("self_improving_report_writer.generation")


class ReportGenerator:
    """Synthesizes structured research reports grounded in retrieved sources."""

    def __init__(
        self,
        llm_service: LLMService,
        config: ReportWriterConfig | None = None,
        provider: str | None = None,
    ) -> None:
        """Initialize generator with LLMService and configuration."""
        self.llm_service = llm_service
        self.config = config or get_default_config()
        self.provider = provider or self.config.generator_provider

    async def generate_report(
        self,
        topic: str,
        sources: Sequence[SourceReference],
        guidelines: list[str] | None = None,
    ) -> tuple[Report, int, float]:
        """Generate structured report from topic and sources.

        Args:
            topic: Research question or topic.
            sources: Sequence of authoritative source chunks.
            guidelines: Optional authoring guidelines (e.g. from past memory).

        Returns:
            Tuple of (Report, tokens_used, cost_usd).
        """
        trimmed_topic = topic.strip()
        if not trimmed_topic:
            raise GenerationError("Topic cannot be empty")

        prompt = build_report_generation_prompt(
            topic=trimmed_topic, sources=sources, guidelines=guidelines
        )

        safe_log(
            _logger,
            logging.INFO,
            f"Generating initial report for topic: '{trimmed_topic[:60]}...'",
            event=ObservabilityEvent.LLM_REQUEST_START,
            provider=self.provider,
            sources_count=len(sources),
        )

        try:
            response = await self.llm_service.complete_structure(
                prompt=prompt,
                response_schema=Report,
                # provider=self.provider,
            )
        except Exception as exc:
            safe_log(
                _logger,
                logging.ERROR,
                f"LLM generation request failed: {exc}",
                event=ObservabilityEvent.LLM_REQUEST_FAILED,
                error=str(exc),
            )
            raise GenerationError(f"LLM generation failed: {exc}") from exc

        if response.error is not None:
            safe_log(
                _logger,
                logging.ERROR,
                f"LLM returned error response: {response.error.message}",
                event=ObservabilityEvent.LLM_REQUEST_FAILED,
                error=response.error.message,
            )
            raise GenerationError(f"LLM generation error: {response.error.message}")

        # Extract token and cost usage
        usage = getattr(response, "usage", {}) or {}
        if isinstance(usage, dict):
            in_tok = int(usage.get("input_tokens", 0))
            out_tok = int(usage.get("output_tokens", 0))
            tokens_used = int(usage.get("total_tokens", in_tok + out_tok))
        else:
            tokens_used = int(getattr(usage, "total_tokens", 0))

        cost_val = getattr(response, "cost_usd", None)
        if cost_val is None and isinstance(getattr(response, "raw_response", None), dict):
            cost_val = response.raw_response.get("cost") or response.raw_response.get("cost_usd")
        cost_usd = float(cost_val or 0.0)

        # Unwrap structured candidate
        candidate = getattr(response, "structured_response", None)
        if candidate is None and hasattr(response, "content"):
            candidate = response.content

        report: Report
        if isinstance(candidate, Report):
            report = candidate
        elif isinstance(candidate, dict):
            try:
                report = Report.model_validate(candidate)
            except Exception as val_exc:
                raise GenerationError(
                    f"Report validation failed on dict response: {val_exc}"
                ) from val_exc
        elif hasattr(response, "text") and response.text:
            try:
                report = Report.model_validate_json(response.text)
            except Exception as json_exc:
                raise GenerationError(
                    f"Report JSON validation failed on raw text response: {json_exc}"
                ) from json_exc
        else:
            raise GenerationError(
                f"Unexpected response structure from LLMService: {type(candidate).__name__}"
            )

        safe_log(
            _logger,
            logging.INFO,
            f"Report generated successfully: '{report.title}'",
            event=ObservabilityEvent.LLM_REQUEST_SUCCESS,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            sections_count=len(report.sections),
        )

        return report, tokens_used, cost_usd
