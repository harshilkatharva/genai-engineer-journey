"""Rubric-based report judge evaluating multi-dimensional quality with model diversity safeguards."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from genai_core import ObservabilityEvent, safe_log

from self_improving_report_writer.core.config import ReportWriterConfig
from self_improving_report_writer.core.exceptions import JudgingError
from self_improving_report_writer.judging.prompts import (
    RUBRIC_JUDGE_SYSTEM_PROMPT,
    LLMJudgeEvaluation,
    build_rubric_judge_prompt,
)
from self_improving_report_writer.models.report import Report, SourceReference
from self_improving_report_writer.models.rubric import (
    DimensionScore,
    ReportJudgeVerdict,
    RubricConfig,
    get_default_rubric_config,
)
from self_improving_report_writer.models.verification import DeterministicVerificationReport

if TYPE_CHECKING:
    from genai_core import LLMService

logger = logging.getLogger(__name__)


class RubricJudge:
    """Evaluates reports across calibrated dimensions using a distinct, high-capacity model with self-judging fallback."""

    def __init__(
        self,
        llm_service: LLMService,
        config: ReportWriterConfig | None = None,
        rubric_config: RubricConfig | None = None,
        judge_model: str | None = None,
        generator_model: str | None = None,
    ) -> None:
        self.llm_service = llm_service
        self.config = config or ReportWriterConfig()
        self.rubric_config = rubric_config or get_default_rubric_config()
        self.judge_model = judge_model or self.config.judge_model
        self.generator_model = generator_model or self.config.generator_model

    async def evaluate(
        self,
        report: Report,
        sources: dict[str, SourceReference],
        deterministic_report: DeterministicVerificationReport | None = None,
    ) -> ReportJudgeVerdict:
        """Evaluate a research report across all configured rubric dimensions."""
        base_threshold = self.rubric_config.acceptance_threshold

        # Check model diversity
        is_self_judged = self.judge_model == self.generator_model
        effective_threshold = base_threshold + (0.05 if is_self_judged else 0.0)
        target_model = self.judge_model

        if is_self_judged:
            safe_log(
                logger,
                logging.WARNING,
                f"Model diversity compromised: Judge model matches generator model ({target_model}). "
                f"Disclosing self-judging and adjusting acceptance threshold from {base_threshold:.2f} to {effective_threshold:.2f}.",
            )

        prompt = build_rubric_judge_prompt(
            report=report,
            rubric_config=self.rubric_config,
            sources=sources,
            deterministic_report=deterministic_report,
        )

        safe_log(
            logger,
            logging.INFO,
            f"Executing rubric evaluation for topic '{report.topic}' using model {target_model} "
            f"(self-judged={is_self_judged}, threshold={effective_threshold:.2f})",
            event=ObservabilityEvent.LLM_REQUEST_START,
        )

        # Call primary judge model
        resp = await self.llm_service.complete_structure(
            prompt=f"{RUBRIC_JUDGE_SYSTEM_PROMPT}\n\n{prompt}",
            response_schema=LLMJudgeEvaluation,
            # provider=target_model,
        )

        # Check for error and fallback if applicable
        if resp.error is not None and not is_self_judged and self.generator_model:
            safe_log(
                logger,
                logging.WARNING,
                f"Primary judge model {target_model} failed with error: {resp.error.message}. "
                f"Falling back to generator model {self.generator_model}.",
            )
            target_model = self.generator_model
            is_self_judged = True
            effective_threshold = base_threshold + 0.05
            resp = await self.llm_service.complete_structure(
                prompt=f"{RUBRIC_JUDGE_SYSTEM_PROMPT}\n\n{prompt}",
                response_schema=LLMJudgeEvaluation,
                # provider=target_model,
            )

        if resp.error is not None:
            raise JudgingError(
                f"Rubric judge execution failed for model {target_model}: {resp.error.message}"
            )

        raw_eval = resp.structured_response
        if raw_eval is None:
            raise JudgingError("Rubric judge returned empty structured response.")

        # Parse raw evaluation
        if isinstance(raw_eval, dict):
            raw_eval = LLMJudgeEvaluation.model_validate(raw_eval)
        elif not isinstance(raw_eval, LLMJudgeEvaluation):
            raise JudgingError(f"Unexpected response type from judge: {type(raw_eval)}")

        # Build dimension scores mapping (case-insensitive lookup)
        raw_scores_map: dict[str, DimensionScore] = {
            d.dimension_name.strip().lower(): d for d in raw_eval.dimension_scores
        }

        # Deterministic mathematical rollup
        final_dim_scores: dict[str, DimensionScore] = {}
        composite_score = 0.0
        all_dimensions_passed = True

        for criterion in self.rubric_config.criteria:
            key = criterion.name.strip().lower()
            if key not in raw_scores_map:
                safe_log(
                    logger,
                    logging.WARNING,
                    f"Judge omitted score for required dimension '{criterion.name}'. Defaulting to 0.0.",
                )
                dim_score_val = 0.0
                dim_reasoning = f"Dimension '{criterion.name}' was not evaluated by the judge."
            else:
                dim_obj = raw_scores_map[key]
                dim_score_val = max(0.0, min(1.0, float(dim_obj.score)))
                dim_reasoning = dim_obj.reasoning

            dim_passed = dim_score_val >= criterion.min_pass_score
            if not dim_passed:
                all_dimensions_passed = False

            final_dim_scores[criterion.name] = DimensionScore(
                dimension_name=criterion.name,
                score=round(dim_score_val, 4),
                passed=dim_passed,
                reasoning=dim_reasoning,
            )
            composite_score += criterion.weight * dim_score_val

        composite_score = round(composite_score, 4)
        passed = (composite_score >= effective_threshold) and all_dimensions_passed

        # Track token usage and costs
        tokens_used = 0
        if resp.usage:
            if isinstance(resp.usage, dict):
                tokens_used = resp.usage.get("total_tokens") or (
                    resp.usage.get("input_tokens", 0) + resp.usage.get("output_tokens", 0)
                )
            else:
                tokens_used = getattr(resp.usage, "total_tokens", 0) or (
                    getattr(resp.usage, "input_tokens", 0) + getattr(resp.usage, "output_tokens", 0)
                )

        cost_usd = 0.0
        if getattr(resp, "cost_usd", None) is not None:
            cost_usd = float(resp.cost_usd)
        elif isinstance(resp.raw_response, dict) and "cost" in resp.raw_response:
            cost_usd = float(resp.raw_response["cost"])

        verdict = ReportJudgeVerdict(
            composite_score=composite_score,
            passed=passed,
            dimension_scores=final_dim_scores,
            overall_feedback=raw_eval.overall_feedback,
            is_self_judged=is_self_judged,
            judge_model=target_model,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
        )

        safe_log(
            logger,
            logging.INFO,
            f"Rubric evaluation complete for '{report.topic}': composite={verdict.composite_score:.4f}, "
            f"passed={verdict.passed} (threshold={effective_threshold:.2f}, self-judged={verdict.is_self_judged})",
            event=ObservabilityEvent.LLM_REQUEST_SUCCESS,
        )

        return verdict
