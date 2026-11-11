"""CandidateSelector: deterministic hierarchical ranking and selection of report candidates."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from self_improving_report_writer.core.config import ReportWriterConfig, get_default_config
from self_improving_report_writer.models.outcome import ReportCandidate

logger = logging.getLogger(__name__)

__all__ = [
    "CandidateSelector",
    "ReportCandidate",
    "compare_candidates",
    "is_candidate_regression",
    "rank_candidates",
    "select_best_candidate",
]


class CandidateSelector:
    """Deterministic hierarchical comparator ranking candidates across all reflection rounds."""

    def __init__(
        self,
        config: ReportWriterConfig | None = None,
        *,
        score_precision: int = 2,
        prefer_parsimony: bool = True,
    ) -> None:
        if score_precision < 0:
            raise ValueError(f"score_precision must be non-negative (got {score_precision})")
        self.config = config or get_default_config()
        self.score_precision = score_precision
        self.prefer_parsimony = prefer_parsimony

    def _quality_key(
        self, candidate: ReportCandidate
    ) -> tuple[int, int, int, int, float, float, float]:
        """Build a strictly transitive quality key across Tiers 1-6 (higher is better)."""
        precision = self.score_precision
        return (
            1 if candidate.deterministic_report.passed else 0,
            -candidate.total_critical_count,
            -candidate.total_major_count,
            1 if candidate.grounding_report.all_claims_grounded else 0,
            round(candidate.grounding_report.grounding_score, precision),
            round(candidate.judge_verdict.composite_score, precision),
            round(candidate.grounding_subscore, precision),
        )

    def ranking_key(
        self, candidate: ReportCandidate
    ) -> tuple[int, int, int, int, float, float, float, int]:
        """Build a strictly transitive lexicographical ranking key across Tiers 1-7 (higher is better)."""
        parsimony_key = -candidate.round_number if self.prefer_parsimony else 0
        return (*self._quality_key(candidate), parsimony_key)

    def compare(self, c1: ReportCandidate, c2: ReportCandidate) -> int:
        """Hierarchical comparison of two candidates using strict weak ordering.

        Returns:
            Positive int (>0) if c1 is strictly better than c2.
            Negative int (<0) if c2 is strictly better than c1.
            Zero (0) if c1 and c2 are tied on all criteria.
        """
        k1 = self.ranking_key(c1)
        k2 = self.ranking_key(c2)
        if k1 > k2:
            return 1
        if k1 < k2:
            return -1
        return 0

    def select_best(self, candidates: Sequence[ReportCandidate]) -> ReportCandidate:
        """Select the highest-ranking candidate from history using hierarchical comparison."""
        if not candidates:
            raise ValueError("Cannot select best candidate from empty candidate list")

        return max(candidates, key=self.ranking_key)

    def rank_candidates(self, candidates: Sequence[ReportCandidate]) -> list[ReportCandidate]:
        """Rank candidates from best to worst using the transitive ranking key."""
        if not candidates:
            return []

        return sorted(candidates, key=self.ranking_key, reverse=True)

    def is_regression(
        self, previous_candidate: ReportCandidate, current_candidate: ReportCandidate
    ) -> bool:
        """Check whether current candidate is strictly worse in quality than previous candidate.

        Note: Quality criteria (Tiers 1-6) determine regression. Parsimony tie-breaker (Tier 7)
        is excluded because current candidate round number is always greater than previous.
        """
        return self._quality_key(current_candidate) < self._quality_key(previous_candidate)


def compare_candidates(c1: ReportCandidate, c2: ReportCandidate) -> int:
    """Compare two candidates using default hierarchical selection settings."""
    return CandidateSelector().compare(c1, c2)


def select_best_candidate(candidates: Sequence[ReportCandidate]) -> ReportCandidate:
    """Select the highest-ranking candidate using default hierarchical selection settings."""
    return CandidateSelector().select_best(candidates)


def rank_candidates(candidates: Sequence[ReportCandidate]) -> list[ReportCandidate]:
    """Rank candidates from best to worst using default hierarchical selection settings."""
    return CandidateSelector().rank_candidates(candidates)


def is_candidate_regression(
    previous_candidate: ReportCandidate, current_candidate: ReportCandidate
) -> bool:
    """Check whether current_candidate regressed relative to previous_candidate."""
    return CandidateSelector().is_regression(previous_candidate, current_candidate)
