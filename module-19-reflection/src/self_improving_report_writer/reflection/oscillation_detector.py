"""OscillationDetector: detects cyclical revision regressions and critique stagnation via SHA-256 fingerprinting."""

from __future__ import annotations

import hashlib
import logging
from typing import Literal

from genai_core import ObservabilityEvent, safe_log
from pydantic import BaseModel, Field

from self_improving_report_writer.models.critique import Critique
from self_improving_report_writer.models.report import Report

logger = logging.getLogger(__name__)


class OscillationCheckResult(BaseModel):
    """Result of cyclical loop and stagnation detection check."""

    has_oscillation: bool = Field(description="Whether a revision loop or stagnation was detected")
    oscillation_type: Literal["none", "content_cycle", "critique_stagnation"] = Field(
        default="none", description="Classification of detected oscillation anomaly"
    )
    details: str = Field(default="", description="Descriptive rationale of the oscillation finding")
    content_fingerprint: str = Field(description="SHA-256 content hash of the current report")
    critique_fingerprint: str = Field(description="SHA-256 critique hash of the current critique")


class OscillationDetector:
    """Tracks SHA-256 fingerprints of report content and critique issue sets across rounds."""

    def __init__(self) -> None:
        self.content_history: list[str] = []
        self.critique_history: list[str] = []

    def reset(self) -> None:
        """Clear fingerprint histories for a fresh report run."""
        self.content_history.clear()
        self.critique_history.clear()

    def check_oscillation(
        self,
        round_number: int,
        report: Report,
        critique: Critique,
    ) -> OscillationCheckResult:
        """Evaluate current candidate and critique against historical fingerprints.

        Args:
            round_number: Current reflection round (0-indexed or 1-indexed).
            report: The report candidate to fingerprint.
            critique: The critique generated for this round.

        Returns:
            OscillationCheckResult indicating whether an oscillation or stagnation occurred.
        """
        content_hash = self.compute_content_fingerprint(report)
        critique_hash = self.compute_critique_fingerprint(critique)

        # 1. Check Content Oscillation Cycle
        # If current content hash was seen in any earlier round:
        if content_hash in self.content_history:
            prev_round_idx = self.content_history.index(content_hash)
            details = (
                f"Content oscillation cycle detected in round {round_number}: report text is identical "
                f"to candidate generated in round {prev_round_idx}. Terminating reflection loop."
            )
            safe_log(
                logger,
                logging.WARNING,
                details,
                event=ObservabilityEvent.TOOL_LOOP_DETECTED,
            )
            self.content_history.append(content_hash)
            self.critique_history.append(critique_hash)
            return OscillationCheckResult(
                has_oscillation=True,
                oscillation_type="content_cycle",
                details=details,
                content_fingerprint=content_hash,
                critique_fingerprint=critique_hash,
            )

        # 2. Check Critique Stagnation
        # If current critique issues exactly match the immediately preceding round:
        if self.critique_history and critique_hash == self.critique_history[-1]:
            details = (
                f"Critique stagnation detected in round {round_number}: the issue set is identical "
                f"to the previous round. Revision failed to resolve identified defects. Terminating reflection loop."
            )
            safe_log(
                logger,
                logging.WARNING,
                details,
                event=ObservabilityEvent.TOOL_LOOP_DETECTED,
            )
            self.content_history.append(content_hash)
            self.critique_history.append(critique_hash)
            return OscillationCheckResult(
                has_oscillation=True,
                oscillation_type="critique_stagnation",
                details=details,
                content_fingerprint=content_hash,
                critique_fingerprint=critique_hash,
            )

        # No oscillation detected: record hashes
        self.content_history.append(content_hash)
        self.critique_history.append(critique_hash)

        return OscillationCheckResult(
            has_oscillation=False,
            oscillation_type="none",
            details="No oscillation or critique stagnation detected.",
            content_fingerprint=content_hash,
            critique_fingerprint=critique_hash,
        )

    @classmethod
    def compute_content_fingerprint(cls, report: Report) -> str:
        """Compute deterministic SHA-256 hash of normalized title and section contents.

        Hash_content = SHA256(normalized_title + sum(normalized_section_content))
        """
        all_sections = [report.executive_summary, *report.sections, report.conclusion]

        norm_title = cls._normalize_text(report.title)
        section_texts = [
            f"{sec.section_id}:{cls._normalize_text(sec.content)}" for sec in all_sections
        ]

        payload = norm_title + "||" + "||".join(section_texts)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def compute_critique_fingerprint(cls, critique: Critique) -> str:
        """Compute deterministic SHA-256 hash of sorted issue taxonomy elements.

        Hash_critique = SHA256(sorted({category + affected_claim_id + severity}))
        """
        if not critique.issues:
            return hashlib.sha256(b"NO_ISSUES").hexdigest()

        signatures = [
            f"{issue.category}:{issue.affected_claim_id or 'none'}:{issue.severity}"
            for issue in critique.issues
        ]
        # Strict deterministic sort
        sorted_payload = "||".join(sorted(signatures))
        return hashlib.sha256(sorted_payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text by lowercasing and collapsing whitespace."""
        return " ".join(text.lower().split())
