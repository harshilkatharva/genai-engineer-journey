"""Reflection package: critique generation, oscillation detection, candidate selection, and revision loop."""

from self_improving_report_writer.reflection.candidate_selector import (
    CandidateSelector,
    ReportCandidate,
)
from self_improving_report_writer.reflection.critique_engine import CritiqueEngine
from self_improving_report_writer.reflection.orchestrator import (
    ReflectionLoopOrchestrator,
    ReflectionRunResult,
)
from self_improving_report_writer.reflection.oscillation_detector import (
    OscillationCheckResult,
    OscillationDetector,
)
from self_improving_report_writer.reflection.revision_engine import RevisionEngine

__all__ = [
    "CandidateSelector",
    "CritiqueEngine",
    "OscillationCheckResult",
    "OscillationDetector",
    "ReflectionLoopOrchestrator",
    "ReflectionRunResult",
    "ReportCandidate",
    "RevisionEngine",
]
