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
    SELF_JUDGING_THRESHOLD_PENALTY,
    ClaimGroundingResult,
    DimensionScore,
    ReportJudgeVerdict,
    RubricConfig,
    RubricCriterion,
    SemanticGroundingReport,
    compute_effective_threshold,
    get_default_rubric_config,
    resolve_effective_threshold,
)
from .verification import (
    DefectCategory,
    DefectSeverity,
    DeterministicVerificationReport,
    PassageMatchResult,
    VerificationIssue,
)

__all__ = [
    "SELF_JUDGING_THRESHOLD_PENALTY",
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
    "compute_effective_threshold",
    "get_default_rubric_config",
    "resolve_effective_threshold",
]
