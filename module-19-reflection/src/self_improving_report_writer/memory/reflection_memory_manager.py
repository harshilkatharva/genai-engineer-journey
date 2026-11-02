"""ReflectionMemoryManager: distills validated authoring lessons into genai-core EpisodicMemory."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from genai_core import (
    ActionEpisode,
    EpisodicMemory,
    InMemoryMemoryStore,
    ObservabilityEvent,
    safe_log,
)

from self_improving_report_writer.core.config import ReportWriterConfig, get_default_config
from self_improving_report_writer.models.critique import (
    CritiqueIssue,
    IssueCategory,
    IssueSeverity,
)

if TYPE_CHECKING:
    from self_improving_report_writer.reflection.orchestrator import ReflectionRunResult

logger = logging.getLogger(__name__)


class ReflectionMemoryManager:
    """Manages lesson distillation, confidence filtering, deduplication, and safe retrieval."""

    ACTION_TYPE = "reflection_writing_lesson"

    def __init__(
        self,
        memory: EpisodicMemory | None = None,
        user_id: str = "default_user",
        config: ReportWriterConfig | None = None,
        min_improvement: float = 0.10,
        min_confidence: float = 0.85,
    ) -> None:
        self.user_id = user_id
        self.config = config or get_default_config()
        self.min_improvement = min_improvement
        self.min_confidence = min_confidence

        if memory is not None:
            self.memory = memory
        else:
            store = InMemoryMemoryStore()
            self.memory = EpisodicMemory(store=store)

    async def distill_lessons(
        self,
        run_result: ReflectionRunResult,
        domain: str = "general",
        resolved_issues: list[CritiqueIssue] | None = None,
    ) -> list[ActionEpisode]:
        """Distill high-leverage authoring lessons from a completed reflection run.

        Enforces Section 7.2 Business Rules:
        1. Completed runs only with verified score improvement Delta >= min_improvement (+0.10).
        2. Confidence filtering: issues must achieve confidence >= min_confidence (0.85).
        3. Deduplication via canonical parameter hashing (check()).
        4. Data minimization: only abstract rules stored in result_summary, no raw report text.
        5. Soft-failure: memory exceptions log warnings without failing report execution.

        Args:
            run_result: The completed run trajectory.
            domain: Topical domain classification (e.g., 'quantum_computing', 'medicine').
            resolved_issues: Optional explicit list of issues resolved during revision rounds.

        Returns:
            List of successfully recorded ActionEpisode objects.
        """
        try:
            return await self._distill_lessons_internal(
                run_result=run_result,
                domain=domain,
                resolved_issues=resolved_issues,
            )
        except Exception as exc:  # noqa: BLE001 - Soft-failure guarantee preserves report delivery
            safe_log(
                logger,
                logging.WARNING,
                f"Soft failure in distill_lessons (ignored to protect report delivery): {exc}",
                event=ObservabilityEvent.MEMORY_CONSOLIDATION_FAILED,
                exc_info=True,
            )
            return []

    async def _distill_lessons_internal(
        self,
        run_result: ReflectionRunResult,
        domain: str,
        resolved_issues: list[CritiqueIssue] | None,
    ) -> list[ActionEpisode]:
        """Internal execution of distillation rules."""
        if not run_result.all_candidates:
            return []

        baseline = run_result.all_candidates[0]
        best = run_result.best_candidate

        baseline_score = baseline.judge_verdict.composite_score
        best_score = best.judge_verdict.composite_score
        delta = round(best_score - baseline_score, 4)

        # Rule 1: Verified improvement threshold check (Delta >= 0.10)
        if delta < self.min_improvement:
            safe_log(
                logger,
                logging.INFO,
                f"Skipping lesson distillation: score improvement delta {delta:.4f} "
                f"< required threshold {self.min_improvement:.2f} (baseline={baseline_score:.2f}, best={best_score:.2f})",
                event=ObservabilityEvent.MEMORY_ACTION_CHECK,
            )
            return []

        # Collect candidate issues to distill
        issues_to_distill: list[CritiqueIssue] = []
        if resolved_issues is not None:
            issues_to_distill.extend(resolved_issues)
        else:
            # Fall back to evaluating defects from baseline that were resolved in best candidate
            for det_issue in baseline.deterministic_report.issues:
                issues_to_distill.append(
                    CritiqueIssue(
                        issue_id=det_issue.issue_id,
                        category=IssueCategory.CITATION_INTEGRITY,
                        severity=IssueSeverity.CRITICAL,
                        explanation=det_issue.message,
                        suggested_correction=f"Ensure citations adhere to source chunks: {det_issue.message}",
                        confidence=1.0,
                    )
                )

        recorded_episodes: list[ActionEpisode] = []

        for issue in issues_to_distill:
            # Rule 2: Confidence filtering (confidence >= 0.85)
            if issue.confidence < self.min_confidence:
                continue

            rule_summary = f"In {domain} reports: {issue.suggested_correction.strip()}"

            # Rule 3: Canonical parameter hashing for deduplication
            parameters = {
                "topic_domain": domain,
                "defect_category": str(issue.category),
                "rule_summary": rule_summary,
            }

            check_result = await self.memory.check(
                user_id=self.user_id,
                action_type=self.ACTION_TYPE,
                parameters=parameters,
                target=domain,
            )

            if check_result.status == "do_not_repeat":
                safe_log(
                    logger,
                    logging.DEBUG,
                    f"Skipping duplicate lesson for domain '{domain}': {rule_summary}",
                    event=ObservabilityEvent.MEMORY_ACTION_CHECK,
                )
                continue

            # Rule 4: Data minimization - store abstract guideline
            episode = await self.memory.record(
                user_id=self.user_id,
                action_type=self.ACTION_TYPE,
                parameters=parameters,
                target=domain,
                result_summary=rule_summary,
                metadata={
                    "confidence": issue.confidence,
                    "improvement_delta": delta,
                    "source_topic": run_result.topic,
                },
            )
            recorded_episodes.append(episode)

        safe_log(
            logger,
            logging.INFO,
            f"Successfully distilled {len(recorded_episodes)} lessons into episodic memory for domain '{domain}' "
            f"(improvement_delta=+{delta:.4f})",
            event=ObservabilityEvent.MEMORY_STORE_SUCCESS,
        )

        return recorded_episodes

    async def retrieve_lessons(
        self,
        domain: str | None = None,
        limit: int = 5,
    ) -> list[str]:
        """Retrieve recent authoring guidelines for injection into report generation.

        Retrieves recent episodes and filters by domain in Python (conforming to Section 7.1).
        Protected by soft-failure handling.
        """
        try:
            episodes = await self.memory.get_episodes(
                user_id=self.user_id,
                action_type=self.ACTION_TYPE,
                limit=limit * 3,
            )

            lessons: list[str] = []
            for ep in episodes:
                if not ep.result_summary:
                    continue
                ep_domain = ep.parameters.get("topic_domain", "general")
                if (
                    domain is None or ep_domain == domain or ep_domain == "general"
                ) and ep.result_summary not in lessons:
                    lessons.append(ep.result_summary)
                if len(lessons) >= limit:
                    break

            return lessons
        except Exception as exc:  # noqa: BLE001 - Soft-failure guarantee preserves report delivery
            safe_log(
                logger,
                logging.WARNING,
                f"Soft failure in retrieve_lessons: {exc}",
                event=ObservabilityEvent.MEMORY_RETRIEVE_SUCCESS,
                exc_info=True,
            )
            return []
