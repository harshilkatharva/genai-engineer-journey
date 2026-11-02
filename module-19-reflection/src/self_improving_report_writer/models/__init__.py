"""Domain models for self-improving report writer."""

from .critique import (
    Critique,
    CritiqueIssue,
    IssueCategory,
    IssueSeverity,
)
from .outcome import (
    AcceptanceResult,
    HumanReviewDecision,
    HumanReviewOutcome,
    HumanReviewPacket,
    ReportStatus,
)
from .report import (
    Citation,
    CitationType,
    Claim,
    Report,
    ReportSection,
    SourceReference,
)
from .rubric import (
    ClaimGroundingResult,
    DimensionScore,
    ReportJudgeVerdict,
    RubricConfig,
    RubricCriterion,
    SemanticGroundingReport,
    get_default_rubric_config,
)
from .verification import (
    DefectCategory,
    DefectSeverity,
    DeterministicVerificationReport,
    PassageMatchResult,
    VerificationIssue,
)

__all__ = [
    "AcceptanceResult",
    "Citation",
    "CitationType",
    "Claim",
    "ClaimGroundingResult",
    "Critique",
    "CritiqueIssue",
    "DefectCategory",
    "DefectSeverity",
    "DeterministicVerificationReport",
    "DimensionScore",
    "HumanReviewDecision",
    "HumanReviewOutcome",
    "HumanReviewPacket",
    "IssueCategory",
    "IssueSeverity",
    "PassageMatchResult",
    "Report",
    "ReportJudgeVerdict",
    "ReportSection",
    "ReportStatus",
    "RubricConfig",
    "RubricCriterion",
    "SemanticGroundingReport",
    "SourceReference",
    "VerificationIssue",
    "get_default_rubric_config",
]
