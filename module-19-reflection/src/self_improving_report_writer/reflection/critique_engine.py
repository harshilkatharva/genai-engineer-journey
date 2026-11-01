"""CritiqueEngine: translates verification defects and rubric deficits into prioritized actionable critiques."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from genai_core import ObservabilityEvent, safe_log

from self_improving_report_writer.models.critique import (
    Critique,
    CritiqueIssue,
    IssueCategory,
    IssueSeverity,
)
from self_improving_report_writer.models.verification import (
    DefectCategory,
    DefectSeverity,
    DeterministicVerificationReport,
)

if TYPE_CHECKING:
    from self_improving_report_writer.models.report import Report
    from self_improving_report_writer.models.rubric import (
        ReportJudgeVerdict,
        SemanticGroundingReport,
    )

logger = logging.getLogger(__name__)


class CritiqueEngine:
    """Translates deterministic failures, semantic grounding defects, and rubric deficits into structured critiques."""

    DEFECT_CATEGORY_MAP: ClassVar[dict[DefectCategory, IssueCategory]] = {
        DefectCategory.SCHEMA: IssueCategory.SCHEMA,
        DefectCategory.CITATION_INTEGRITY: IssueCategory.CITATION_INTEGRITY,
        DefectCategory.FACTUAL_GROUNDING: IssueCategory.FACTUAL_GROUNDING,
        DefectCategory.COMPLETENESS: IssueCategory.COMPLETENESS,
        DefectCategory.ORGANIZATION_CLARITY: IssueCategory.CLARITY_ORGANIZATION,
    }

    SEVERITY_MAP: ClassVar[dict[DefectSeverity, IssueSeverity]] = {
        DefectSeverity.CRITICAL: IssueSeverity.CRITICAL,
        DefectSeverity.MAJOR: IssueSeverity.MAJOR,
        DefectSeverity.MINOR: IssueSeverity.MINOR,
    }

    DIMENSION_CATEGORY_MAP: ClassVar[dict[str, IssueCategory]] = {
        "factual_accuracy": IssueCategory.FACTUAL_GROUNDING,
        "evidence_grounding": IssueCategory.FACTUAL_GROUNDING,
        "citation_correctness": IssueCategory.CITATION_INTEGRITY,
        "completeness": IssueCategory.COMPLETENESS,
        "relevance": IssueCategory.RELEVANCE,
        "clarity_organization": IssueCategory.CLARITY_ORGANIZATION,
        "logical_consistency": IssueCategory.LOGICAL_CONSISTENCY,
    }

    def generate_critique(
        self,
        round_number: int,
        report: Report,
        deterministic_report: DeterministicVerificationReport | None = None,
        grounding_report: SemanticGroundingReport | None = None,
        judge_verdict: ReportJudgeVerdict | None = None,
        acceptance_threshold: float = 0.85,
    ) -> Critique:
        """Aggregate verification findings and rubric judge feedback into a structured Critique.

        Args:
            round_number: Current reflection loop round index.
            report: The report candidate evaluated.
            deterministic_report: Gate 1 deterministic verification report.
            grounding_report: Gate 2 semantic grounding report.
            judge_verdict: Rubric judge verdict and dimension scores.
            acceptance_threshold: Minimum composite score required for acceptance.

        Returns:
            Structured Critique with prioritized issues and revision directive.
        """
        issues: list[CritiqueIssue] = []
        issue_counter = 1

        # 1. Translate Gate 1 Deterministic Verification Defects
        if deterministic_report and deterministic_report.issues:
            for d_issue in deterministic_report.issues:
                cat = self.DEFECT_CATEGORY_MAP.get(
                    d_issue.category, IssueCategory.CITATION_INTEGRITY
                )
                sev = self.SEVERITY_MAP.get(d_issue.severity, IssueSeverity.MAJOR)
                correction = self._suggest_deterministic_correction(d_issue.message, cat)

                issues.append(
                    CritiqueIssue(
                        issue_id=f"CRITIQUE-{issue_counter:03d}",
                        category=cat,
                        severity=sev,
                        affected_section_id=d_issue.section_id,
                        affected_claim_id=d_issue.claim_id,
                        explanation=d_issue.message,
                        suggested_correction=correction,
                        confidence=1.0,
                    )
                )
                issue_counter += 1

        # 2. Translate Gate 2 Semantic Grounding Defects
        if grounding_report and grounding_report.issues:
            for g_issue in grounding_report.issues:
                cat = self.DEFECT_CATEGORY_MAP.get(
                    g_issue.category, IssueCategory.FACTUAL_GROUNDING
                )
                sev = self.SEVERITY_MAP.get(g_issue.severity, IssueSeverity.MAJOR)
                correction = self._suggest_grounding_correction(g_issue.message)

                issues.append(
                    CritiqueIssue(
                        issue_id=f"CRITIQUE-{issue_counter:03d}",
                        category=cat,
                        severity=sev,
                        affected_section_id=g_issue.section_id,
                        affected_claim_id=g_issue.claim_id,
                        explanation=g_issue.message,
                        suggested_correction=correction,
                        confidence=0.95,
                    )
                )
                issue_counter += 1

        # 3. Translate Rubric Judge Dimension Deficits
        if judge_verdict and judge_verdict.dimension_scores:
            for dim_name, dim_score in judge_verdict.dimension_scores.items():
                if not dim_score.passed or dim_score.score < 0.80:
                    cat = self.DIMENSION_CATEGORY_MAP.get(
                        dim_name.lower(), IssueCategory.CLARITY_ORGANIZATION
                    )
                    sev = self._determine_rubric_deficit_severity(dim_name, dim_score.score)
                    correction = self._suggest_rubric_correction(dim_name, dim_score.reasoning)

                    issues.append(
                        CritiqueIssue(
                            issue_id=f"CRITIQUE-{issue_counter:03d}",
                            category=cat,
                            severity=sev,
                            explanation=(
                                f"Rubric dimension '{dim_name}' scored {dim_score.score:.2f} "
                                f"(below acceptable threshold): {dim_score.reasoning}"
                            ),
                            suggested_correction=correction,
                            confidence=0.90,
                        )
                    )
                    issue_counter += 1

        # Deduplicate and sort issues: CRITICAL > MAJOR > MINOR
        prioritized_issues = self._prioritize_and_deduplicate(issues)

        critical_count = sum(1 for i in prioritized_issues if i.severity == IssueSeverity.CRITICAL)
        major_count = sum(1 for i in prioritized_issues if i.severity == IssueSeverity.MAJOR)
        minor_count = sum(1 for i in prioritized_issues if i.severity == IssueSeverity.MINOR)

        # Decide should_revise strictly according to Business Logic Section 4.2
        composite_score = judge_verdict.composite_score if judge_verdict else 0.0
        grounding_score = grounding_report.grounding_score if grounding_report else 1.0
        eff_threshold = (
            (acceptance_threshold + 0.05)
            if (judge_verdict and judge_verdict.is_self_judged)
            else acceptance_threshold
        )

        should_revise = self._determine_should_revise(
            critical_count=critical_count,
            major_count=major_count,
            minor_count=minor_count,
            composite_score=composite_score,
            grounding_score=grounding_score,
            effective_threshold=eff_threshold,
        )

        summary = self._build_critique_summary(
            round_number=round_number,
            critical_count=critical_count,
            major_count=major_count,
            minor_count=minor_count,
            composite_score=composite_score,
            should_revise=should_revise,
        )

        critique = Critique(
            round_number=round_number,
            issues=prioritized_issues,
            summary=summary,
            should_revise=should_revise,
        )

        safe_log(
            logger,
            logging.INFO,
            f"Critique generated for round {round_number}: issues={len(prioritized_issues)} "
            f"(crit={critical_count}, maj={major_count}, min={minor_count}), should_revise={should_revise}",
            event=ObservabilityEvent.LLM_REQUEST_SUCCESS,
        )

        return critique

    def _determine_should_revise(
        self,
        critical_count: int,
        major_count: int,
        minor_count: int,
        composite_score: float,
        grounding_score: float,
        effective_threshold: float,
    ) -> bool:
        """Enforce Section 4.2 Severity-Based Revision Thresholds."""
        # 1. Critical Issues Present -> Always triggers revision
        if critical_count > 0:
            return True

        # 2. Major Issues Present -> Triggers revision if composite score < threshold or grounding < 0.85
        if major_count > 0:
            return (composite_score < effective_threshold) or (grounding_score < 0.85)

        # 3. Minor Issues Only (zero critical, zero major) ->
        # If composite score >= threshold, loop terminates immediately, accepting the report.
        if composite_score >= effective_threshold:
            return False

        # If composite score < threshold even with only minor issues, revise to improve score
        return minor_count > 0

    def _determine_rubric_deficit_severity(
        self, dimension_name: str, score: float
    ) -> IssueSeverity:
        """Assign issue severity to rubric dimension deficit."""
        normalized_name = dimension_name.lower().strip()
        if normalized_name == "factual_accuracy" and score < 0.85:
            return IssueSeverity.CRITICAL
        if normalized_name in {"evidence_grounding", "citation_correctness"} and score < 0.85:
            return IssueSeverity.MAJOR
        if score < 0.70:
            return IssueSeverity.MAJOR
        return IssueSeverity.MINOR

    def _suggest_deterministic_correction(self, message: str, category: IssueCategory) -> str:
        """Generate concrete actionable correction for deterministic verification failure."""
        lower_msg = message.lower()
        if "quotation" in lower_msg and ("missing" in lower_msg or "substring" in lower_msg):
            return (
                "Ensure quoted passages are exact verbatim substrings of the referenced source chunk, "
                "or reclassify the citation as PARAPHRASE without quotation marks."
            )
        if "unknown" in lower_msg or "not present" in lower_msg:
            return (
                "Reference only source_id and chunk_id values that exist in the application-controlled "
                "retrieved source collection. Do not fabricate identifiers."
            )
        if "lacks supporting citations" in lower_msg:
            return "Attach at least one valid Citation to the factual claim."
        if "prose" in lower_msg:
            return "Include citation markers like [cite_id] in the section prose to link citations."
        return f"Correct structural or citation defect: {message}"

    def _suggest_grounding_correction(self, message: str) -> str:
        """Generate concrete actionable correction for semantic grounding failure."""
        lower_msg = message.lower()
        if "contradicts" in lower_msg:
            return (
                "The factual claim refutes cited source evidence. Revise the claim to accurately reflect "
                "the source findings, or remove the contradicted statement entirely."
            )
        if "weakly supported" in lower_msg or "neutral" in lower_msg:
            return (
                "The cited source does not substantiate the specific claim. Either cite a source that directly "
                "proves the proposition, or soften the assertion to match the available evidence."
            )
        if "ungrounded factual assertion in prose" in lower_msg:
            return (
                "Convert the substantive factual statement in prose into a formal structured Claim with citations, "
                "or rephrase it as a non-factual executive synthesis."
            )
        return "Ensure all factual propositions are directly entailed by verified source chunks."

    def _suggest_rubric_correction(self, dimension_name: str, reasoning: str) -> str:
        """Generate concrete actionable correction based on rubric dimension and reasoning."""
        dim = dimension_name.lower()
        if "factual_accuracy" in dim:
            return f"Eliminate unsubstantiated claims and align with sources: {reasoning}"
        if "evidence_grounding" in dim:
            return f"Strengthen evidence linkage and eliminate ungrounded statements: {reasoning}"
        if "citation_correctness" in dim:
            return f"Ensure all citations precisely map to verified chunk evidence: {reasoning}"
        if "completeness" in dim:
            return f"Broaden coverage to address missing aspects or study limitations: {reasoning}"
        if "relevance" in dim:
            return f"Remove off-topic padding and focus tightly on research prompt: {reasoning}"
        if "clarity_organization" in dim:
            return (
                f"Enhance section hierarchy, executive tone, and narrative synthesis: {reasoning}"
            )
        if "logical_consistency" in dim:
            return f"Resolve contradictory statements between sections: {reasoning}"
        return f"Improve {dimension_name}: {reasoning}"

    def _prioritize_and_deduplicate(self, issues: list[CritiqueIssue]) -> list[CritiqueIssue]:
        """Deduplicate issues by explanation and sort by severity (CRITICAL > MAJOR > MINOR)."""
        severity_rank = {
            IssueSeverity.CRITICAL: 0,
            IssueSeverity.MAJOR: 1,
            IssueSeverity.MINOR: 2,
        }

        seen_signatures: set[str] = set()
        unique_issues: list[CritiqueIssue] = []

        for issue in issues:
            sig = f"{issue.category}:{issue.affected_claim_id or ''}:{issue.explanation.strip().lower()}"
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                unique_issues.append(issue)

        unique_issues.sort(key=lambda i: severity_rank[i.severity])
        return unique_issues

    def _build_critique_summary(
        self,
        round_number: int,
        critical_count: int,
        major_count: int,
        minor_count: int,
        composite_score: float,
        should_revise: bool,
    ) -> str:
        """Build executive summary of reflection critique."""
        if not should_revise:
            return (
                f"Round {round_number} Critique: Report meets acceptance standards "
                f"(composite_score={composite_score:.2f}, critical={critical_count}, major={major_count}). "
                "No further revisions required."
            )
        return (
            f"Round {round_number} Critique: Revision required. "
            f"Detected {critical_count} critical, {major_count} major, and {minor_count} minor deficiencies. "
            f"Current composite score: {composite_score:.2f}. Focus revision on critical and major evidence fixes."
        )
