"""Multi-stage verification gate integrating Module 19 deterministic and semantic verifiers."""

from __future__ import annotations

from typing import Any

from genai_core import LLMService
from genai_core.observability import get_logger
from pydantic import BaseModel, ConfigDict, Field
from self_improving_report_writer.models.rubric import SemanticGroundingReport
from self_improving_report_writer.models.verification import DeterministicVerificationReport
from self_improving_report_writer.verification.deterministic_verifier import DeterministicVerifier
from self_improving_report_writer.verification.semantic_grounding_verifier import (
    SemanticGroundingVerifier,
)

from ..core.config import ResearchTeamConfig, get_research_team_config
from ..models import Report
from .source_pool import AuthoritativeSourcePool

_logger = get_logger("parallel_research_team.synthesis.verifier")


class SynthesisVerificationResult(BaseModel):
    """Consolidated findings and pass/fail verdict from verification gates."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    passed: bool = Field(description="True if all enabled verification gates passed")
    gate1_passed: bool = Field(description="Gate 1 structural and citation integrity outcome")
    gate2_passed: bool | None = Field(
        default=None, description="Gate 2 semantic claim entailment outcome if enabled"
    )
    gate1_report: DeterministicVerificationReport | None = None
    gate2_report: SemanticGroundingReport | None = None
    issues: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = Field(default="")


class ResearchVerifier:
    """Enforces two-stage verification on synthesized reports.

    Gate 1: Pure-Python deterministic structural, citation pointer, and quotation substring checks.
    Gate 2: Model-based semantic claim-passage entailment verification calibrated at threshold 0.70.
    """

    def __init__(
        self,
        config: ResearchTeamConfig | None = None,
        judge_llm_service: LLMService | None = None,
        deterministic_verifier: DeterministicVerifier | None = None,
    ) -> None:
        self.config = config or get_research_team_config()
        self.judge_llm_service = judge_llm_service or LLMService(
            default_provider=self.config.judge_provider,
            default_model=self.config.judge_model,
        )
        self.deterministic_verifier = deterministic_verifier or DeterministicVerifier()

    async def verify(
        self,
        report: Report,
        source_pool: AuthoritativeSourcePool,
    ) -> SynthesisVerificationResult:
        """Run Gate 1 and Gate 2 verification against the authoritative source pool."""
        sources_list = source_pool.list_sources()

        # ---------------------------------------------------------------------
        # Gate 1: Deterministic Verification
        # ---------------------------------------------------------------------
        _logger.info("Executing Gate 1 deterministic verification on report '%s'", report.title)
        gate1_report = self.deterministic_verifier.verify(report, sources_list)
        gate1_issues = [i.model_dump() for i in gate1_report.issues]

        if not gate1_report.passed:
            _logger.warning(
                "Gate 1 deterministic verification failed with %d critical, %d major issues",
                gate1_report.critical_count,
                gate1_report.major_count,
            )
            return SynthesisVerificationResult(
                passed=False,
                gate1_passed=False,
                gate2_passed=None,
                gate1_report=gate1_report,
                issues=gate1_issues,
                summary=(
                    f"Gate 1 failed: {gate1_report.critical_count} critical and "
                    f"{gate1_report.major_count} major citation/structural defects found."
                ),
            )

        # ---------------------------------------------------------------------
        # Gate 2: Semantic Grounding Verification (Optional / Configurable)
        # ---------------------------------------------------------------------
        if not self.config.enable_semantic_verification:
            _logger.info("Gate 2 semantic verification disabled by configuration")
            return SynthesisVerificationResult(
                passed=True,
                gate1_passed=True,
                gate2_passed=None,
                gate1_report=gate1_report,
                issues=gate1_issues,
                summary="Gate 1 passed (Gate 2 semantic verification disabled).",
            )

        _logger.info(
            "Executing Gate 2 semantic grounding verification with threshold=%.2f using %s/%s",
            self.config.semantic_verification_threshold,
            self.config.judge_provider,
            self.config.judge_model,
        )

        semantic_verifier = SemanticGroundingVerifier(
            llm_service=self.judge_llm_service,
            entailment_threshold=self.config.semantic_verification_threshold,
        )

        try:
            gate2_report = await semantic_verifier.verify_grounding(report, sources_list)
        except Exception as exc:
            _logger.exception("Gate 2 semantic verification encountered an error")
            return SynthesisVerificationResult(
                passed=False,
                gate1_passed=True,
                gate2_passed=False,
                gate1_report=gate1_report,
                issues=gate1_issues + [{"error": f"Gate 2 judge execution error: {exc}"}],
                summary=f"Gate 2 semantic verification failed due to judge error: {exc}",
            )

        gate2_issues = [i.model_dump() for i in gate2_report.issues]
        all_issues = gate1_issues + gate2_issues
        passed = bool(gate1_report.passed and gate2_report.passed)

        summary = (
            f"All verification gates passed (Gate 2 score: {gate2_report.grounding_score:.2f})."
            if passed
            else (
                f"Gate 2 semantic verification failed (Grounding score: {gate2_report.grounding_score:.2f} < "
                f"threshold {self.config.semantic_verification_threshold:.2f})."
            )
        )

        return SynthesisVerificationResult(
            passed=passed,
            gate1_passed=gate1_report.passed,
            gate2_passed=gate2_report.passed,
            gate1_report=gate1_report,
            gate2_report=gate2_report,
            issues=all_issues,
            summary=summary,
        )
