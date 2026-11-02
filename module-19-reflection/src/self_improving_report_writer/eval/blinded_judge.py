"""Blinded judge implementation ensuring unbiased evaluation across approaches."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from self_improving_report_writer.judging.rubric_judge import RubricJudge
from self_improving_report_writer.models.report import Report, SourceReference

if TYPE_CHECKING:
    from genai_core import LLMService

    from self_improving_report_writer.core.config import ReportWriterConfig
    from self_improving_report_writer.models.rubric import ReportJudgeVerdict
    from self_improving_report_writer.models.verification import (
        DeterministicVerificationReport,
    )

logger = logging.getLogger(__name__)


class BlindedJudge:
    """Evaluates anonymized report candidates to eliminate evaluation bias."""

    def __init__(
        self,
        rubric_judge: RubricJudge | None = None,
        llm_service: LLMService | None = None,
        config: ReportWriterConfig | None = None,
    ) -> None:
        self.rubric_judge = rubric_judge
        self.llm_service = llm_service
        self.config = config

    def anonymize_report(self, report: Report) -> Report:
        """Strip all approach provenance, round numbers, and metadata from a report.

        Args:
            report: The candidate report to anonymize.

        Returns:
            A clean Report clone with provenance metadata stripped.
        """
        return Report(
            topic=report.topic,
            title=report.title,
            executive_summary=report.executive_summary,
            sections=report.sections,
            conclusion=report.conclusion,
            limitations=report.limitations,
            version=1,
            created_by="anonymized_candidate",
        )

    async def evaluate_blinded(
        self,
        report: Report,
        sources: dict[str, SourceReference] | list[SourceReference],
        deterministic_report: DeterministicVerificationReport | None = None,
    ) -> ReportJudgeVerdict:
        """Perform blinded rubric evaluation on an anonymized report candidate.

        Args:
            report: The report candidate to evaluate.
            sources: Authoritative source references.
            deterministic_report: Optional deterministic report for citation fact-checking.

        Returns:
            ReportJudgeVerdict containing calibrated dimension and composite scores.
        """
        if self.rubric_judge is None:
            from genai_core import LLMService

            service = self.llm_service or LLMService()
            self.rubric_judge = RubricJudge(llm_service=service, config=self.config)

        if isinstance(sources, list):
            sources_dict = {s.chunk_id: s for s in sources}
        else:
            sources_dict = sources

        anonymized = self.anonymize_report(report)
        return await self.rubric_judge.evaluate(
            report=anonymized,
            sources=sources_dict,
            deterministic_report=deterministic_report,
        )
