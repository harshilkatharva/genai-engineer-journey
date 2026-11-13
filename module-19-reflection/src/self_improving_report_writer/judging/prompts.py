"""Prompts and response schemas for rubric judging and semantic grounding verification."""

from __future__ import annotations

from typing import TYPE_CHECKING

from self_improving_report_writer.models.llm_schemas import (
    ClaimEntailmentEvaluation,
    LLMJudgeEvaluation,
    ProseConsistencyEvaluation,
    SectionClaimsEntailmentEvaluation,
    SingleClaimEntailmentVerdict,
)
from self_improving_report_writer.prompts.manager import (
    escape_xml_context,
    get_prompt_manager,
)

if TYPE_CHECKING:
    from self_improving_report_writer.models.report import Report, SourceReference
    from self_improving_report_writer.models.rubric import RubricConfig
    from self_improving_report_writer.models.verification import DeterministicVerificationReport

_prompt_mgr = get_prompt_manager()

RUBRIC_JUDGE_SYSTEM_PROMPT = _prompt_mgr.get_system_prompt("judging", "system")
CLAIM_ENTAILMENT_SYSTEM_PROMPT = _prompt_mgr.get_system_prompt(
    "verification", "claim_entailment_system"
)
PROSE_CONSISTENCY_SYSTEM_PROMPT = _prompt_mgr.get_system_prompt(
    "verification", "prose_consistency_system"
)


def build_rubric_judge_prompt(
    report: Report,
    rubric_config: RubricConfig,
    sources: dict[str, SourceReference],
    deterministic_report: DeterministicVerificationReport | None = None,
) -> str:
    """Build user prompt for the rubric judge evaluation."""
    return get_prompt_manager().render_rubric_judge_prompt(
        report=report,
        rubric_config=rubric_config,
        sources=sources,
        deterministic_report=deterministic_report,
    )


def build_claim_entailment_prompt(
    claim_id: str,
    claim_text: str,
    cited_chunks: list[tuple[str, str]],
) -> str:
    """Build prompt to evaluate semantic entailment between a claim and its cited chunk passages."""
    return get_prompt_manager().render_claim_entailment_prompt(
        claim_id=claim_id,
        claim_text=claim_text,
        cited_chunks=cited_chunks,
    )


def build_batch_claim_entailment_prompt(
    section_id: str,
    section_title: str,
    claims: list[tuple[str, str, list[str]]],
    cited_chunks: list[tuple[str, str]],
) -> str:
    """Build prompt to evaluate semantic entailment for multiple claims in a section against deduplicated source chunks."""
    return get_prompt_manager().render_batch_claim_entailment_prompt(
        section_id=section_id,
        section_title=section_title,
        claims=claims,
        cited_chunks=cited_chunks,
    )


def build_prose_consistency_prompt(
    section_id: str,
    section_title: str,
    prose_content: str,
    structured_claims: list[str],
) -> str:
    """Build prompt to evaluate whether section prose contains factual statements missing from structured claims."""
    return get_prompt_manager().render_prose_consistency_prompt(
        section_id=section_id,
        section_title=section_title,
        prose_content=prose_content,
        structured_claims=structured_claims,
    )


__all__ = [
    "CLAIM_ENTAILMENT_SYSTEM_PROMPT",
    "PROSE_CONSISTENCY_SYSTEM_PROMPT",
    "RUBRIC_JUDGE_SYSTEM_PROMPT",
    "ClaimEntailmentEvaluation",
    "LLMJudgeEvaluation",
    "ProseConsistencyEvaluation",
    "SectionClaimsEntailmentEvaluation",
    "SingleClaimEntailmentVerdict",
    "build_batch_claim_entailment_prompt",
    "build_claim_entailment_prompt",
    "build_prose_consistency_prompt",
    "build_rubric_judge_prompt",
    "escape_xml_context",
]
