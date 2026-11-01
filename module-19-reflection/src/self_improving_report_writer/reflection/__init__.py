"""Reflection package: critique generation, oscillation detection, and revision loop."""

from self_improving_report_writer.reflection.critique_engine import CritiqueEngine
from self_improving_report_writer.reflection.oscillation_detector import (
    OscillationCheckResult,
    OscillationDetector,
)

__all__ = [
    "CritiqueEngine",
    "OscillationCheckResult",
    "OscillationDetector",
]
