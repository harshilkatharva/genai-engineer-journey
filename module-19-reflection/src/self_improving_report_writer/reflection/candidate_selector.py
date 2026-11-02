"""CandidateSelector: deterministic hierarchical ranking and selection of report candidates."""

from __future__ import annotations

import functools
import logging

from self_improving_report_writer.models.outcome import ReportCandidate

logger = logging.getLogger(__name__)

__all__ = ["CandidateSelector", "ReportCandidate"]


class CandidateSelector:
    """Deterministic hierarchical comparator ranking candidates across all reflection rounds."""

    @classmethod
    def compare(cls, c1: ReportCandidate, c2: ReportCandidate) -> int:
        """Hierarchical comparison of two candidates.

        Returns:
            Positive int (>0) if c1 is strictly better than c2.
            Negative int (<0) if c2 is strictly better than c1.
            Zero (0) if c1 and c2 are tied on all criteria.
        """
        # Tier 1: Gate 1 Deterministic Verification (Pass strictly beats Fail)
        if c1.deterministic_report.passed != c2.deterministic_report.passed:
            return 1 if c1.deterministic_report.passed else -1

        # Tier 2: Total Critical Defect Count (Fewer strictly beats More)
        if c1.total_critical_count != c2.total_critical_count:
            return 1 if c1.total_critical_count < c2.total_critical_count else -1

        # Tier 3: Total Major Defect Count (Fewer strictly beats More)
        if c1.total_major_count != c2.total_major_count:
            return 1 if c1.total_major_count < c2.total_major_count else -1

        # Tier 4: Semantic Grounding Status (All claims grounded, then grounding score)
        if c1.grounding_report.all_claims_grounded != c2.grounding_report.all_claims_grounded:
            return 1 if c1.grounding_report.all_claims_grounded else -1

        diff_grounding = round(
            c1.grounding_report.grounding_score - c2.grounding_report.grounding_score, 4
        )
        if abs(diff_grounding) >= 0.01:
            return 1 if diff_grounding > 0 else -1

        # Tier 5: Composite Rubric Score (Descending)
        diff_composite = round(
            c1.judge_verdict.composite_score - c2.judge_verdict.composite_score, 4
        )
        if abs(diff_composite) >= 0.01:
            return 1 if diff_composite > 0 else -1

        # Tier 6: Grounding Sub-scores (factual_accuracy + evidence_grounding)
        diff_subscore = round(c1.grounding_subscore - c2.grounding_subscore, 4)
        if abs(diff_subscore) >= 0.01:
            return 1 if diff_subscore > 0 else -1

        # Tier 7 (Tie-breaker): Parsimony (Earlier round preferred to conserve token expenditure)
        if c1.round_number != c2.round_number:
            return 1 if c1.round_number < c2.round_number else -1

        return 0

    @classmethod
    def select_best(cls, candidates: list[ReportCandidate]) -> ReportCandidate:
        """Select the highest-ranking candidate from history using hierarchical comparison."""
        if not candidates:
            raise ValueError("Cannot select best candidate from empty candidate list")

        best = candidates[0]
        for candidate in candidates[1:]:
            if cls.compare(candidate, best) > 0:
                best = candidate
        return best

    @classmethod
    def rank_candidates(cls, candidates: list[ReportCandidate]) -> list[ReportCandidate]:
        """Rank candidates from best to worst using the hierarchical comparator."""
        if not candidates:
            return []

        # functools.cmp_to_key with reversed comparison for descending order
        def _cmp(a: ReportCandidate, b: ReportCandidate) -> int:
            return -cls.compare(a, b)

        return sorted(candidates, key=functools.cmp_to_key(_cmp))

    @classmethod
    def is_regression(
        cls, previous_candidate: ReportCandidate, current_candidate: ReportCandidate
    ) -> bool:
        """Check whether current candidate is strictly worse in quality than previous candidate.

        Note: Quality criteria (Tiers 1-6) determine regression. Parsimony tie-breaker (Tier 7)
        is excluded because current candidate round number is always greater than previous.
        """
        # Tier 1: Gate 1 Deterministic Verification
        if (
            current_candidate.deterministic_report.passed
            != previous_candidate.deterministic_report.passed
        ):
            return not current_candidate.deterministic_report.passed

        # Tier 2: Total Critical Defect Count
        if current_candidate.total_critical_count != previous_candidate.total_critical_count:
            return current_candidate.total_critical_count > previous_candidate.total_critical_count

        # Tier 3: Total Major Defect Count
        if current_candidate.total_major_count != previous_candidate.total_major_count:
            return current_candidate.total_major_count > previous_candidate.total_major_count

        # Tier 4: Semantic Grounding Status
        if (
            current_candidate.grounding_report.all_claims_grounded
            != previous_candidate.grounding_report.all_claims_grounded
        ):
            return not current_candidate.grounding_report.all_claims_grounded

        diff_grounding = round(
            current_candidate.grounding_report.grounding_score
            - previous_candidate.grounding_report.grounding_score,
            4,
        )
        if abs(diff_grounding) >= 0.01:
            return diff_grounding < 0

        # Tier 5: Composite Rubric Score
        diff_composite = round(
            current_candidate.judge_verdict.composite_score
            - previous_candidate.judge_verdict.composite_score,
            4,
        )
        if abs(diff_composite) >= 0.01:
            return diff_composite < 0

        # Tier 6: Grounding Sub-scores
        diff_subscore = round(
            current_candidate.grounding_subscore - previous_candidate.grounding_subscore, 4
        )
        if abs(diff_subscore) >= 0.01:
            return diff_subscore < 0

        return False
