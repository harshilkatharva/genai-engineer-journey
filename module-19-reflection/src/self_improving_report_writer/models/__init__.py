"""Domain models for self-improving report writer."""

from .report import (
    Citation,
    CitationType,
    Claim,
    Report,
    ReportSection,
    SourceReference,
)
from .verification import (
    DefectCategory,
    DefectSeverity,
    DeterministicVerificationReport,
    PassageMatchResult,
    VerificationIssue,
)

__all__ = [
    "Citation",
    "CitationType",
    "Claim",
    "DefectCategory",
    "DefectSeverity",
    "DeterministicVerificationReport",
    "PassageMatchResult",
    "Report",
    "ReportSection",
    "SourceReference",
    "VerificationIssue",
]
