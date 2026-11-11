"""CandidateSelector: deterministic hierarchical ranking and selection of report candidates."""

from __future__ import annotations

import logging

from self_improving_report_writer.models.outcome import ReportCandidate

logger = logging.getLogger(__name__)

__all__ = ["CandidateSelector", "ReportCandidate"]


class CandidateSelector:
    """Deterministic hierarchical comparator ranking candidates across all reflection rounds."""

    @classmethod
    def _quality_key(
        cls, candidate: ReportCandidate
    ) -> tuple[int, int, int, int, float, float, float]:
        """Build a strictly transitive quality key across Tiers 1-6 (higher is better)."""
        return (
            1 if candidate.deterministic_report.passed else 0,
            -candidate.total_critical_count,
            -candidate.total_major_count,
            1 if candidate.grounding_report.all_claims_grounded else 0,
            round(candidate.grounding_report.grounding_score, 2),
            round(candidate.judge_verdict.composite_score, 2),
            round(candidate.grounding_subscore, 2),
        )

    @classmethod
    def ranking_key(
        cls, candidate: ReportCandidate
    ) -> tuple[int, int, int, int, float, float, float, int]:
        """Build a strictly transitive lexicographical ranking key across Tiers 1-7 (higher is better)."""
        return (*cls._quality_key(candidate), -candidate.round_number)

    @classmethod
    def compare(cls, c1: ReportCandidate, c2: ReportCandidate) -> int:
        """Hierarchical comparison of two candidates using strict weak ordering.

        Returns:
            Positive int (>0) if c1 is strictly better than c2.
            Negative int (<0) if c2 is strictly better than c1.
            Zero (0) if c1 and c2 are tied on all criteria.
        """
        k1 = cls.ranking_key(c1)
        k2 = cls.ranking_key(c2)
        if k1 > k2:
            return 1
        if k1 < k2:
            return -1
        return 0

    @classmethod
    def select_best(cls, candidates: list[ReportCandidate]) -> ReportCandidate:
        """Select the highest-ranking candidate from history using hierarchical comparison."""
        if not candidates:
            raise ValueError("Cannot select best candidate from empty candidate list")

        return max(candidates, key=cls.ranking_key)

    @classmethod
    def rank_candidates(cls, candidates: list[ReportCandidate]) -> list[ReportCandidate]:
        """Rank candidates from best to worst using the transitive ranking key."""
        if not candidates:
            return []

        return sorted(candidates, key=cls.ranking_key, reverse=True)

    @classmethod
    def is_regression(
        cls, previous_candidate: ReportCandidate, current_candidate: ReportCandidate
    ) -> bool:
        """Check whether current candidate is strictly worse in quality than previous candidate.

        Note: Quality criteria (Tiers 1-6) determine regression. Parsimony tie-breaker (Tier 7)
        is excluded because current candidate round number is always greater than previous.
        """
        return cls._quality_key(current_candidate) < cls._quality_key(previous_candidate)
