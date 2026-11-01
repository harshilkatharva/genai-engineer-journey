"""Self-Improving Report Writer package."""

from .models.report import (
    Citation,
    CitationType,
    Claim,
    Report,
    ReportSection,
    SourceReference,
)
from .models.verification import (
    DefectCategory,
    DefectSeverity,
    DeterministicVerificationReport,
    PassageMatchResult,
    VerificationIssue,
)
from .verification.deterministic_verifier import (
    DeterministicVerifier,
    normalize_for_matching,
)

__all__ = [
    "Citation",
    "CitationType",
    "Claim",
    "DefectCategory",
    "DefectSeverity",
    "DeterministicVerificationReport",
    "DeterministicVerifier",
    "PassageMatchResult",
    "Report",
    "ReportSection",
    "SourceReference",
    "VerificationIssue",
    "normalize_for_matching",
]
