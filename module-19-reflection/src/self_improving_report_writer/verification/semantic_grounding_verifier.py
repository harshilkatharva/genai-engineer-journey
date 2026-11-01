"""Gate 2 semantic grounding verification: claim-passage entailment and prose-claim consistency."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from genai_core import ObservabilityEvent, safe_log

from self_improving_report_writer.core.config import ReportWriterConfig
from self_improving_report_writer.core.exceptions import VerificationError
from self_improving_report_writer.judging.prompts import (
    CLAIM_ENTAILMENT_SYSTEM_PROMPT,
    PROSE_CONSISTENCY_SYSTEM_PROMPT,
    ClaimEntailmentEvaluation,
    ProseConsistencyEvaluation,
    build_claim_entailment_prompt,
    build_prose_consistency_prompt,
)
from self_improving_report_writer.models.report import Report, SourceReference
from self_improving_report_writer.models.rubric import ClaimGroundingResult, SemanticGroundingReport
from self_improving_report_writer.models.verification import (
    DefectCategory,
    DefectSeverity,
    VerificationIssue,
)

if TYPE_CHECKING:
    from genai_core import LLMService

logger = logging.getLogger(__name__)


class SemanticGroundingVerifier:
    """Evaluates semantic entailment between claims and cited passages, and checks prose-claim consistency."""

    def __init__(
        self,
        llm_service: LLMService,
        config: ReportWriterConfig | None = None,
        evaluator_model: str | None = None,
        entailment_threshold: float = 0.85,
        check_prose_consistency: bool = True,
    ) -> None:
        self.llm_service = llm_service
        self.config = config or ReportWriterConfig()
        self.evaluator_model = evaluator_model or self.config.judge_model
        self.entailment_threshold = entailment_threshold
        self.check_prose_consistency = check_prose_consistency

    async def verify_grounding(
        self,
        report: Report,
        retrieved_sources: (
            dict[tuple[str, str], SourceReference]
            | dict[str, SourceReference]
            | Sequence[SourceReference]
        ),
    ) -> SemanticGroundingReport:
        """Perform Gate 2 semantic grounding verification.

        Args:
            report: The Report instance to verify.
            retrieved_sources: Application-controlled source pool.

        Returns:
            SemanticGroundingReport with per-claim entailment and defect issues.
        """
        source_map = self._build_source_map(retrieved_sources)
        claim_evaluations: list[ClaimGroundingResult] = []
        issues: list[VerificationIssue] = []

        total_tokens = 0
        total_cost = 0.0
        issue_counter = 1

        all_sections = [report.executive_summary, *report.sections, report.conclusion]

        safe_log(
            logger,
            logging.INFO,
            f"Starting Gate 2 semantic grounding verification for report '{report.title}' using model {self.evaluator_model}",
            event=ObservabilityEvent.LLM_REQUEST_START,
        )

        for sec in all_sections:
            factual_claims = [c for c in sec.claims if c.is_factual]

            for claim in factual_claims:
                if not claim.citations:
                    issues.append(
                        VerificationIssue(
                            issue_id=f"G2-DEFECT-{issue_counter:03d}",
                            category=DefectCategory.FACTUAL_GROUNDING,
                            severity=DefectSeverity.MAJOR,
                            message=f"Factual claim '{claim.claim_id}' lacks supporting citations.",
                            section_id=sec.section_id,
                            claim_id=claim.claim_id,
                        )
                    )
                    issue_counter += 1
                    claim_evaluations.append(
                        ClaimGroundingResult(
                            claim_id=claim.claim_id,
                            section_id=sec.section_id,
                            entailment="NEUTRAL",
                            score=0.0,
                            reasoning="Claim has no citations attached to substantiate the factual proposition.",
                            cited_chunk_ids=[],
                        )
                    )
                    continue

                # Collect cited chunks
                cited_chunks: list[tuple[str, str]] = []
                missing_chunk_ids: list[str] = []

                for cite in claim.citations:
                    source_key = (cite.source_id, cite.chunk_id)
                    s_ref = source_map.get(source_key)
                    if s_ref is None:
                        missing_chunk_ids.append(f"{cite.source_id}:{cite.chunk_id}")
                    else:
                        cited_chunks.append((cite.chunk_id, s_ref.content))

                if not cited_chunks:
                    issues.append(
                        VerificationIssue(
                            issue_id=f"G2-DEFECT-{issue_counter:03d}",
                            category=DefectCategory.FACTUAL_GROUNDING,
                            severity=DefectSeverity.MAJOR,
                            message=(
                                f"Factual claim '{claim.claim_id}' references unknown chunks: {missing_chunk_ids}"
                            ),
                            section_id=sec.section_id,
                            claim_id=claim.claim_id,
                            details={"missing_chunk_ids": missing_chunk_ids},
                        )
                    )
                    issue_counter += 1
                    claim_evaluations.append(
                        ClaimGroundingResult(
                            claim_id=claim.claim_id,
                            section_id=sec.section_id,
                            entailment="NEUTRAL",
                            score=0.0,
                            reasoning=f"Cited chunks could not be resolved from sources: {missing_chunk_ids}",
                            cited_chunk_ids=[],
                        )
                    )
                    continue

                # Run claim entailment check via LLM
                prompt = build_claim_entailment_prompt(
                    claim_id=claim.claim_id,
                    claim_text=claim.text,
                    cited_chunks=cited_chunks,
                )

                resp = await self.llm_service.complete_structure(
                    prompt=prompt,
                    response_schema=ClaimEntailmentEvaluation,
                    system_instruction=CLAIM_ENTAILMENT_SYSTEM_PROMPT,
                    provider=self.evaluator_model,
                )

                if resp.error is not None:
                    raise VerificationError(
                        f"Claim entailment check failed for claim '{claim.claim_id}': {resp.error.message}"
                    )

                if resp.usage:
                    if isinstance(resp.usage, dict):
                        total_tokens += resp.usage.get("total_tokens") or (
                            resp.usage.get("input_tokens", 0) + resp.usage.get("output_tokens", 0)
                        )
                    else:
                        total_tokens += getattr(resp.usage, "total_tokens", 0) or (
                            getattr(resp.usage, "input_tokens", 0)
                            + getattr(resp.usage, "output_tokens", 0)
                        )
                if getattr(resp, "cost_usd", None) is not None:
                    total_cost += float(resp.cost_usd)
                elif isinstance(resp.raw_response, dict) and "cost" in resp.raw_response:
                    total_cost += float(resp.raw_response["cost"])

                eval_data = resp.structured_response
                if isinstance(eval_data, dict):
                    eval_data = ClaimEntailmentEvaluation.model_validate(eval_data)
                elif not isinstance(eval_data, ClaimEntailmentEvaluation):
                    raise VerificationError(
                        f"Unexpected response type from entailment evaluator: {type(eval_data)}"
                    )

                entailment = eval_data.entailment
                reasoning = eval_data.reasoning
                score = eval_data.score

                # Enforce rubric rules: ENTAILS -> 1.0, NEUTRAL -> 0.5, CONTRADICTS -> 0.0
                if entailment == "ENTAILS":
                    score = 1.0
                elif entailment == "NEUTRAL":
                    score = min(score, 0.5)
                    issues.append(
                        VerificationIssue(
                            issue_id=f"G2-DEFECT-{issue_counter:03d}",
                            category=DefectCategory.FACTUAL_GROUNDING,
                            severity=DefectSeverity.MAJOR,
                            message=f"Claim '{claim.claim_id}' is only weakly supported or neutral with respect to cited evidence: {reasoning}",
                            section_id=sec.section_id,
                            claim_id=claim.claim_id,
                            details={"reasoning": reasoning},
                        )
                    )
                    issue_counter += 1
                elif entailment == "CONTRADICTS":
                    score = 0.0
                    issues.append(
                        VerificationIssue(
                            issue_id=f"G2-DEFECT-{issue_counter:03d}",
                            category=DefectCategory.FACTUAL_GROUNDING,
                            severity=DefectSeverity.CRITICAL,
                            message=f"Claim '{claim.claim_id}' contradicts cited source evidence: {reasoning}",
                            section_id=sec.section_id,
                            claim_id=claim.claim_id,
                            details={"reasoning": reasoning},
                        )
                    )
                    issue_counter += 1

                claim_evaluations.append(
                    ClaimGroundingResult(
                        claim_id=claim.claim_id,
                        section_id=sec.section_id,
                        entailment=entailment,
                        score=score,
                        reasoning=reasoning,
                        cited_chunk_ids=[c_id for c_id, _ in cited_chunks],
                    )
                )

            # Check prose-claim consistency
            if self.check_prose_consistency and len(sec.content.strip()) > 30:
                structured_claim_texts = [c.text for c in sec.claims]
                prose_prompt = build_prose_consistency_prompt(
                    section_id=sec.section_id,
                    section_title=sec.title,
                    prose_content=sec.content,
                    structured_claims=structured_claim_texts,
                )

                p_resp = await self.llm_service.complete_structure(
                    prompt=prose_prompt,
                    response_schema=ProseConsistencyEvaluation,
                    system_instruction=PROSE_CONSISTENCY_SYSTEM_PROMPT,
                    provider=self.evaluator_model,
                )

                if p_resp.error is None and p_resp.structured_response:
                    if p_resp.usage:
                        if isinstance(p_resp.usage, dict):
                            total_tokens += p_resp.usage.get("total_tokens") or (
                                p_resp.usage.get("input_tokens", 0)
                                + p_resp.usage.get("output_tokens", 0)
                            )
                        else:
                            total_tokens += getattr(p_resp.usage, "total_tokens", 0) or (
                                getattr(p_resp.usage, "input_tokens", 0)
                                + getattr(p_resp.usage, "output_tokens", 0)
                            )

                    if getattr(p_resp, "cost_usd", None) is not None:
                        total_cost += float(p_resp.cost_usd)
                    elif isinstance(p_resp.raw_response, dict) and "cost" in p_resp.raw_response:
                        total_cost += float(p_resp.raw_response["cost"])

                    p_eval = p_resp.structured_response
                    if isinstance(p_eval, dict):
                        p_eval = ProseConsistencyEvaluation.model_validate(p_eval)

                    if isinstance(p_eval, ProseConsistencyEvaluation):
                        for unstr_claim in p_eval.unstructured_claims:
                            issues.append(
                                VerificationIssue(
                                    issue_id=f"G2-DEFECT-{issue_counter:03d}",
                                    category=DefectCategory.FACTUAL_GROUNDING,
                                    severity=DefectSeverity.MAJOR,
                                    message=(
                                        f"Section '{sec.section_id}' contains ungrounded factual assertion "
                                        f"in prose not captured as a claim: '{unstr_claim}'"
                                    ),
                                    section_id=sec.section_id,
                                    details={"unstructured_claim": unstr_claim},
                                )
                            )
                            issue_counter += 1

        crit_count = sum(1 for i in issues if i.severity == DefectSeverity.CRITICAL)
        maj_count = sum(1 for i in issues if i.severity == DefectSeverity.MAJOR)
        min_count = sum(1 for i in issues if i.severity == DefectSeverity.MINOR)

        grounding_score = (
            round(sum(c.score for c in claim_evaluations) / len(claim_evaluations), 4)
            if claim_evaluations
            else 1.0
        )
        all_grounded = (
            all(c.entailment == "ENTAILS" for c in claim_evaluations) if claim_evaluations else True
        )
        passed = (
            (crit_count == 0)
            and (maj_count == 0)
            and (grounding_score >= self.entailment_threshold)
        )

        report_out = SemanticGroundingReport(
            passed=passed,
            all_claims_grounded=all_grounded,
            grounding_score=grounding_score,
            claim_evaluations=claim_evaluations,
            issues=issues,
            critical_count=crit_count,
            major_count=maj_count,
            minor_count=min_count,
            tokens_used=total_tokens,
            cost_usd=round(total_cost, 6),
        )

        safe_log(
            logger,
            logging.INFO,
            f"Gate 2 semantic grounding complete: passed={report_out.passed}, "
            f"grounding_score={report_out.grounding_score:.4f} (crit={crit_count}, maj={maj_count})",
            event=ObservabilityEvent.LLM_REQUEST_SUCCESS,
        )

        return report_out

    def _build_source_map(
        self,
        retrieved_sources: (
            dict[tuple[str, str], SourceReference]
            | dict[str, SourceReference]
            | Sequence[SourceReference]
        ),
    ) -> dict[tuple[str, str], SourceReference]:
        """Normalize retrieved source collection into a (source_id, chunk_id) lookup dict."""
        source_map: dict[tuple[str, str], SourceReference] = {}

        if isinstance(retrieved_sources, dict):
            for k, v in retrieved_sources.items():
                if isinstance(k, tuple) and len(k) == 2:
                    source_map[(str(k[0]), str(k[1]))] = v
                else:
                    source_map[(v.source_id, v.chunk_id)] = v
        else:
            for s in retrieved_sources:
                source_map[(s.source_id, s.chunk_id)] = s

        return source_map
