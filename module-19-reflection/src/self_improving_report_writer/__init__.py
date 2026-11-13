"""Self-Improving Report Writer package."""

from self_improving_report_writer.core.config import ReportWriterConfig, get_default_config
from self_improving_report_writer.core.exceptions import (
    BudgetExceededError,
    GenerationError,
    ReflectionError,
    ReportWriterError,
    RetrievalError,
    VerificationError,
)
from self_improving_report_writer.generation.prompts import (
    build_report_generation_prompt,
    format_source_contexts,
)
from self_improving_report_writer.generation.report_generator import ReportGenerator
from self_improving_report_writer.models.report import (
    Citation,
    CitationType,
    Claim,
    Report,
    ReportSection,
    SourceReference,
)
from self_improving_report_writer.models.verification import (
    DefectCategory,
    DefectSeverity,
    DeterministicVerificationReport,
    PassageMatchResult,
    VerificationIssue,
)
from self_improving_report_writer.retrieval.source_retriever import SourceRetriever
from self_improving_report_writer.verification.deterministic_verifier import (
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
