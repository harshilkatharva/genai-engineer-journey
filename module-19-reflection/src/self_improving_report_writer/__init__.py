"""Self-Improving Report Writer package."""

from .core.config import ReportWriterConfig, get_default_config
from .core.exceptions import (
    BudgetExceededError,
    GenerationError,
    ReflectionError,
    ReportWriterError,
    RetrievalError,
    VerificationError,
)
from .generation.prompts import (
    build_report_generation_prompt,
    format_source_contexts,
)
from .generation.report_generator import ReportGenerator
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
from .retrieval.source_retriever import SourceRetriever
from .verification.deterministic_verifier import (
    DeterministicVerifier,
    normalize_for_matching,
)

__all__ = [
    "BudgetExceededError",
    "Citation",
    "CitationType",
    "Claim",
    "DefectCategory",
    "DefectSeverity",
    "DeterministicVerificationReport",
    "DeterministicVerifier",
    "GenerationError",
    "PassageMatchResult",
    "ReflectionError",
    "Report",
    "ReportGenerator",
    "ReportSection",
    "ReportWriterConfig",
    "ReportWriterError",
    "RetrievalError",
    "SourceReference",
    "SourceRetriever",
    "VerificationError",
    "VerificationIssue",
    "build_report_generation_prompt",
    "format_source_contexts",
    "get_default_config",
    "normalize_for_matching",
]
