"""Self-correcting reflection engine for Gate 2 verification defects."""

from __future__ import annotations

from typing import Any

from genai_core import LLMService
from genai_core.observability import get_logger
from pydantic import BaseModel, Field

from ..core.config import ResearchTeamConfig, get_research_team_config
from ..models import Report
from ..prompts.manager import PromptTemplateManager, get_prompt_manager
from ..telemetry.pricing import PricingCalculator
from .source_pool import AuthoritativeSourcePool
from .verifier import ResearchVerifier, SynthesisVerificationResult

_logger = get_logger("parallel_research_team.synthesis.reflection")


class ReflectionOutcome(BaseModel):
    """Result of self-correcting reflection process."""

    report: Report = Field(description="Best report candidate after reflection")
    verification_result: SynthesisVerificationResult = Field(
        description="Verification outcome of the returned report"
    )
    steps_executed: int = Field(default=0, ge=0, description="Reflection steps executed")
    tokens_used: int = Field(default=0, ge=0, description="Tokens consumed across reflection")
    cost_usd: float = Field(default=0.0, ge=0.0, description="USD cost incurred across reflection")
    improved: bool = Field(default=False, description="Whether reflection improved report quality")
    exhausted_budget: bool = Field(
        default=False, description="Whether reflection budget was exhausted"
    )


class ReflectionEngine:
    """Revises synthesized research reports targeting Gate 2 claim entailment defects.

    Operates under an isolated token and financial envelope so that reflection spend
    never aborts or exhausts base research execution.
    """

    def __init__(
        self,
        config: ResearchTeamConfig | None = None,
        llm_service: LLMService | None = None,
        prompt_manager: PromptTemplateManager | None = None,
        pricing_calculator: PricingCalculator | None = None,
    ) -> None:
        self.config = config or get_research_team_config()
        self.llm_service = llm_service or LLMService(
            default_provider=self.config.synthesis_provider,
            default_model=self.config.synthesis_model,
        )
        self.prompt_manager = prompt_manager or get_prompt_manager(self.config)
        self.pricing_calculator = pricing_calculator or PricingCalculator()

    async def reflect_and_revise(
        self,
        topic: str,
        current_report: Report,
        source_pool: AuthoritativeSourcePool,
        verification_result: SynthesisVerificationResult,
        verifier: ResearchVerifier,
    ) -> ReflectionOutcome:
        """Execute iterative self-correction on ungrounded claims until pass or budget ceiling."""
        if verification_result.passed or not self.config.enable_reflection:
            return ReflectionOutcome(
                report=current_report,
                verification_result=verification_result,
                steps_executed=0,
                tokens_used=0,
                cost_usd=0.0,
                improved=False,
                exhausted_budget=False,
            )

        max_steps = self.config.max_reflection_steps
        max_tokens = self.config.max_reflection_tokens
        max_cost_usd = self.config.max_reflection_cost_usd

        best_report = current_report
        best_verification = verification_result
        accumulated_tokens = 0
        accumulated_cost_usd = 0.0
        steps_executed = 0
        improved = False
        exhausted_budget = False

        _logger.info(
            "Starting reflection loop for '%s' (max_steps=%d, token_envelope=%d, cost_envelope=$%.4f)",
            topic,
            max_steps,
            max_tokens,
            max_cost_usd,
        )

        for step in range(1, max_steps + 1):
            # Check isolated reflection budget before step
            if accumulated_tokens >= max_tokens or accumulated_cost_usd >= max_cost_usd:
                _logger.warning(
                    "Reflection budget exhausted at step %d: tokens=%d/%d, cost=$%.5f/$%.4f",
                    step,
                    accumulated_tokens,
                    max_tokens,
                    accumulated_cost_usd,
                    max_cost_usd,
                )
                exhausted_budget = True
                break

            # 1. Extract defective claims from Gate 2 report
            defects: list[dict[str, Any]] = []
            claim_text_map: dict[str, str] = {}
            for sec in best_report.sections:
                for c in sec.claims:
                    claim_text_map[c.claim_id] = c.text

            if best_verification.gate2_report:
                for eval_item in best_verification.gate2_report.claim_evaluations:
                    if eval_item.entailment in ("NEUTRAL", "CONTRADICTS") or eval_item.score < 0.70:
                        defects.append(
                            {
                                "claim_id": eval_item.claim_id,
                                "section_id": eval_item.section_id,
                                "claim_text": claim_text_map.get(eval_item.claim_id, ""),
                                "entailment": eval_item.entailment,
                                "score": eval_item.score,
                                "reasoning": eval_item.reasoning,
                                "cited_chunk_ids": eval_item.cited_chunk_ids,
                            }
                        )

            # If no specific Gate 2 claim defects extracted, collect general issues
            general_issues: list[dict[str, Any]] = []
            for issue in best_verification.issues:
                if isinstance(issue, dict):
                    general_issues.append(issue)

            if not defects and not general_issues:
                _logger.info(
                    "No actionable verification defects found for reflection step %d", step
                )
                break

            steps_executed += 1
            _logger.info(
                "Executing reflection step %d/%d targeting %d claim defects and %d issues",
                step,
                max_steps,
                len(defects),
                len(general_issues),
            )

            # 2. Render reflection prompt
            system_prompt, user_prompt = self.prompt_manager.render_reflection_prompts(
                topic=topic,
                current_report_json=best_report.model_dump_json(indent=2),
                verification_summary=best_verification.summary,
                defects=defects,
                general_issues=general_issues,
                sources=source_pool.list_sources(),
                version=self.config.prompt_version,
            )

            # 3. Call LLM for structured report revision
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            try:
                response = await self.llm_service.complete_structure(
                    prompt=full_prompt,
                    response_schema=Report,
                    provider=self.config.synthesis_provider,
                    model=self.config.synthesis_model,
                )
                err = getattr(response, "error", None)
                if err and isinstance(getattr(err, "message", None), str):
                    _logger.warning("Reflection step %d LLM error: %s", step, err.message)
                    break
                revised_candidate = response.structured_response
                step_tokens = (response.prompt_tokens or 0) + (response.completion_tokens or 0)
                step_cost = self.pricing_calculator.calculate_cost(
                    self.config.synthesis_model,
                    response.prompt_tokens or 0,
                    response.completion_tokens or 0,
                )
            except Exception as exc:  # noqa: BLE001 - non-destructive fallback preserves original report candidate
                _logger.warning("Reflection step %d LLM generation error: %s", step, exc)
                break

            accumulated_tokens += step_tokens
            accumulated_cost_usd += step_cost

            # 4. Re-verify the revised candidate
            _logger.info("Re-verifying revised report candidate from reflection step %d", step)
            new_verification = await verifier.verify(
                report=revised_candidate,
                source_pool=source_pool,
            )
            accumulated_tokens += new_verification.tokens_used
            accumulated_cost_usd += new_verification.cost_usd

            # 5. Determine whether candidate improved
            if new_verification.passed:
                _logger.info(
                    "Reflection step %d successfully resolved all verification issues!", step
                )
                best_report = revised_candidate
                best_verification = new_verification
                improved = True
                break

            # If not passed, check if quality score improved or issues decreased
            new_g2_score = (
                new_verification.gate2_report.grounding_score
                if new_verification.gate2_report
                else 0.0
            )
            prev_g2_score = (
                best_verification.gate2_report.grounding_score
                if best_verification.gate2_report
                else 0.0
            )
            if new_g2_score > prev_g2_score or len(new_verification.issues) < len(
                best_verification.issues
            ):
                _logger.info(
                    "Reflection step %d improved grounding score from %.2f to %.2f",
                    step,
                    prev_g2_score,
                    new_g2_score,
                )
                best_report = revised_candidate
                best_verification = new_verification
                improved = True

        # Append limitation note if verification advisory still persists
        if not best_verification.passed:
            refl_note = (
                f"Reflection attempted ({steps_executed} steps); verification advisory: "
                f"{best_verification.summary}"
            )
            if refl_note not in best_report.limitations:
                best_report.limitations.append(refl_note)
            if refl_note not in best_report.unresolved_limitations:
                best_report.unresolved_limitations.append(refl_note)

        return ReflectionOutcome(
            report=best_report,
            verification_result=best_verification,
            steps_executed=steps_executed,
            tokens_used=accumulated_tokens,
            cost_usd=round(accumulated_cost_usd, 6),
            improved=improved,
            exhausted_budget=exhausted_budget,
        )
