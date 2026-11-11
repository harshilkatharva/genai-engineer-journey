"""Rubric-based report judge evaluating multi-dimensional quality with model diversity safeguards."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from genai_core import ObservabilityEvent, safe_log

from self_improving_report_writer.core.config import ReportWriterConfig
from self_improving_report_writer.core.exceptions import JudgingError
from self_improving_report_writer.core.pricing import extract_or_estimate_cost
from self_improving_report_writer.core.rate_limit_controller import (
    EvaluationRateLimitController,
    get_shared_rate_limit_controller,
)
from self_improving_report_writer.judging.prompts import (
    RUBRIC_JUDGE_SYSTEM_PROMPT,
    build_rubric_judge_prompt,
)
from self_improving_report_writer.models.llm_schemas import LLMJudgeEvaluation
from self_improving_report_writer.models.report import Report, SourceReference
from self_improving_report_writer.models.rubric import (
    DimensionScore,
    ReportJudgeVerdict,
    RubricConfig,
    compute_effective_threshold,
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
        judge_provider: str | None = None,
        rate_limit_controller: EvaluationRateLimitController | None = None,
    ) -> None:
        self.llm_service = llm_service
        self.config = config or ReportWriterConfig()
        if rubric_config is not None:
            self.rubric_config = rubric_config
        elif config is not None:
            self.rubric_config = get_default_rubric_config().model_copy(
                update={"acceptance_threshold": self.config.acceptance_threshold}
            )
        else:
            self.rubric_config = get_default_rubric_config()
        self.judge_model = judge_model or self.config.judge_model
        self.generator_model = generator_model or self.config.generator_model
        self.judge_provider = judge_provider or self.config.judge_provider
        self.rate_limit_controller = rate_limit_controller or get_shared_rate_limit_controller(
            config=self.config
        )

    async def evaluate(
        self,
        report: Report,
        sources: dict[str, SourceReference],
        deterministic_report: DeterministicVerificationReport | None = None,
        deadline: float | None = None,
    ) -> ReportJudgeVerdict:
        """Evaluate a research report across all configured rubric dimensions."""
        base_threshold = self.rubric_config.acceptance_threshold

        # Check model diversity
        is_self_judged = self.judge_model == self.generator_model
        effective_threshold = compute_effective_threshold(
            base_threshold, is_self_judged=is_self_judged
        )
        target_model = self.judge_model
        target_provider = self.judge_provider

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
        resp = await self.rate_limit_controller.complete_structure(
            self.llm_service,
            prompt=f"{RUBRIC_JUDGE_SYSTEM_PROMPT}\n\n{prompt}",
            response_schema=LLMJudgeEvaluation,
            deadline=deadline,
            provider=target_provider,
            model=target_model,
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
            target_provider = self.config.generator_provider
            is_self_judged = True
            effective_threshold = compute_effective_threshold(base_threshold, is_self_judged=True)
            resp = await self.rate_limit_controller.complete_structure(
                self.llm_service,
                prompt=f"{RUBRIC_JUDGE_SYSTEM_PROMPT}\n\n{prompt}",
                response_schema=LLMJudgeEvaluation,
                deadline=deadline,
                provider=target_provider,
                model=target_model,
            )

        if resp.error is not None:
            raise JudgingError(
                f"Rubric judge execution failed for model {target_model}: {resp.error.message}"
            )

        # Runtime model identity verification (Option A)
        actual_model = getattr(resp, "model", None)
        if actual_model and actual_model == self.generator_model and not is_self_judged:
            is_self_judged = True
            effective_threshold = compute_effective_threshold(base_threshold, is_self_judged=True)
            safe_log(
                logger,
                logging.WARNING,
                f"Model diversity check: Returned model '{actual_model}' matches generator model '{self.generator_model}'. "
                f"Engaging self-judging penalty (+0.05) to raise effective threshold to {effective_threshold:.2f}.",
                event=ObservabilityEvent.LLM_REQUEST_START,
            )
        if actual_model:
            target_model = actual_model

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
        _, _, tokens_used, cost_usd = extract_or_estimate_cost(
            response=resp,
            model=target_model,
            default_model=self.judge_model,
        )

        verdict = ReportJudgeVerdict(
            composite_score=composite_score,
            passed=passed,
            dimension_scores=final_dim_scores,
            overall_feedback=raw_eval.overall_feedback,
            is_self_judged=is_self_judged,
            effective_threshold=effective_threshold,
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
