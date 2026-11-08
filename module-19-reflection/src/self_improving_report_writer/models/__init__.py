"""Domain models for self-improving report writer."""

from .critique import (
    Critique,
    CritiqueIssue,
    IssueCategory,
    IssueSeverity,
)
from .llm_schemas import (
    ClaimEntailmentEvaluation,
    LLMJudgeEvaluation,
    ProseConsistencyEvaluation,
    SectionClaimsEntailmentEvaluation,
    SingleClaimEntailmentVerdict,
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
    "ClaimEntailmentEvaluation",
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
    "LLMJudgeEvaluation",
    "PassageMatchResult",
    "ProseConsistencyEvaluation",
    "Report",
    "ReportJudgeVerdict",
    "ReportSection",
    "ReportStatus",
    "RubricConfig",
    "RubricCriterion",
    "SectionClaimsEntailmentEvaluation",
    "SemanticGroundingReport",
    "SingleClaimEntailmentVerdict",
    "SourceReference",
    "VerificationIssue",
    "get_default_rubric_config",
]
