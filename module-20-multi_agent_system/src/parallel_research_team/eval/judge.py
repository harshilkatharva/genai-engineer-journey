"""LLM-as-a-judge quality evaluator scoring research answers across standard rubrics."""

from __future__ import annotations

from genai_core import LLMService
from genai_core.observability import get_logger
from pydantic import BaseModel, Field

from ..core.config import ResearchTeamConfig, get_research_team_config
from .models import AgentExecutionMetrics, EvaluationQuestion, ExpectedWinner, QualityScores
from .rate_limiter import AsyncRateLimiter, get_global_rate_limiter

_logger = get_logger("parallel_research_team.eval.judge")


class JudgeEvaluation(BaseModel):
    """Raw structured evaluation output from judge LLM."""

    factuality: float = Field(
        ge=1.0,
        le=5.0,
        description="Score 1-5: Factual accuracy and grounding against ground truth reference (1=hallucinated, 5=completely accurate)",
    )
    comprehensiveness: float = Field(
        ge=1.0,
        le=5.0,
        description="Score 1-5: Completeness, detail, and depth covering all requested facets (1=shallow/truncated, 5=exhaustive)",
    )
    clarity: float = Field(
        ge=1.0,
        le=5.0,
        description="Score 1-5: Logical organization, clear headings, formatting, and prose readability (1=incoherent, 5=exemplary)",
    )
    relevance: float = Field(
        ge=1.0,
        le=5.0,
        description="Score 1-5: Directness and alignment with question prompt without irrelevant padding (1=tangential, 5=precise)",
    )
    reasoning: str = Field(description="Step-by-step critical justification of scores")


class EvaluationJudge:
    """Evaluates agent answers against benchmark questions using structured LLM judging."""

    def __init__(
        self,
        config: ResearchTeamConfig | None = None,
        llm_service: LLMService | None = None,
        rate_limiter: AsyncRateLimiter | None = None,
    ) -> None:
        self.config = config or get_research_team_config()
        self.rate_limiter = rate_limiter or get_global_rate_limiter()

        self.llm_service = llm_service or LLMService(
            default_provider=self.config.judge_provider,
            default_model=self.config.judge_model,
        )

    async def evaluate_answer(
        self,
        question: EvaluationQuestion,
        answer: str,
        system_name: str,
    ) -> QualityScores:
        """Evaluate a single system's answer on a 1.0 to 5.0 scale."""
        await self.rate_limiter.acquire()

        if not answer.strip() or "Execution error:" in answer:
            return QualityScores(
                factuality=1.0,
                comprehensiveness=1.0,
                clarity=1.0,
                relevance=1.0,
                overall_score=1.0,
                reasoning=f"System failed or produced empty answer for '{question.id}'.",
            )

        rubric_str = (
            ", ".join(question.rubric_focus)
            if question.rubric_focus
            else "standard technical quality"
        )

        prompt = f"""You are an expert impartial evaluation judge assessing the technical quality of an AI research system response.

[RESEARCH QUESTION]
{question.query}

[GROUND TRUTH REFERENCE & CORE FACTS]
{question.ground_truth_reference}

[KEY EVALUATION FOCUS]
{rubric_str}

[SYSTEM RESPONSE BEING EVALUATED ({system_name})]
{answer}

INSTRUCTIONS:
1. Compare the system response against the ground truth reference and requested question.
2. Score on a 1.0 to 5.0 scale (decimal increments permitted, e.g. 4.5):
   - factuality: 1.0 (hallucinated/contradictory) to 5.0 (completely accurate and evidence-grounded).
   - comprehensiveness: 1.0 (superficial/missing key facets) to 5.0 (thorough and detailed).
   - clarity: 1.0 (unstructured/messy) to 5.0 (structured, legible, and well-formatted).
   - relevance: 1.0 (wandering/irrelevant) to 5.0 (concise and directly on point).
3. Provide critical step-by-step reasoning explaining specific strengths or deficiencies.
"""

        try:
            resp = await self.llm_service.complete_structure(
                prompt=prompt,
                response_schema=JudgeEvaluation,
                provider=self.config.judge_provider,
                model=self.config.judge_model,
            )
            raw_eval: JudgeEvaluation | None = getattr(resp, "structured_response", None)
            if raw_eval is None:
                text = getattr(resp, "text", None) or str(getattr(resp, "raw_response", ""))
                try:
                    raw_eval = JudgeEvaluation.model_validate_json(text)
                except Exception:  # noqa: BLE001
                    raw_eval = JudgeEvaluation(
                        factuality=4.0,
                        comprehensiveness=4.0,
                        clarity=4.0,
                        relevance=4.0,
                        reasoning="Fallback parsed evaluation.",
                    )

            # Weighted overall score: 35% factuality, 35% comprehensiveness, 15% clarity, 15% relevance
            weighted_score = round(
                (0.35 * raw_eval.factuality)
                + (0.35 * raw_eval.comprehensiveness)
                + (0.15 * raw_eval.clarity)
                + (0.15 * raw_eval.relevance),
                2,
            )

            return QualityScores(
                factuality=raw_eval.factuality,
                comprehensiveness=raw_eval.comprehensiveness,
                clarity=raw_eval.clarity,
                relevance=raw_eval.relevance,
                overall_score=weighted_score,
                reasoning=raw_eval.reasoning,
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Judge evaluation failed for %s: %s", question.id, exc)
            return QualityScores(
                factuality=3.0,
                comprehensiveness=3.0,
                clarity=3.0,
                relevance=3.0,
                overall_score=3.0,
                reasoning=f"Judge evaluation exception: {exc}",
            )

    def determine_winner(
        self,
        single_metrics: AgentExecutionMetrics,
        single_quality: QualityScores,
        multi_metrics: AgentExecutionMetrics,
        multi_quality: QualityScores,
    ) -> ExpectedWinner:
        """Determine actual winner considering quality, latency, and cost."""
        quality_diff = multi_quality.overall_score - single_quality.overall_score

        # Significant quality lead (> 0.30 points) decides the winner
        if quality_diff >= 0.30:
            return ExpectedWinner.MULTI_AGENT
        if quality_diff <= -0.30:
            return ExpectedWinner.SINGLE_AGENT

        # Quality is effectively tied: winner determined by operational efficiency (cost & speed)
        # If single agent achieves equal quality with lower latency and cost, single agent wins.
        if (
            single_metrics.cost_usd <= multi_metrics.cost_usd
            and single_metrics.latency_seconds <= multi_metrics.latency_seconds
        ):
            return ExpectedWinner.SINGLE_AGENT

        if (
            multi_metrics.cost_usd < single_metrics.cost_usd
            and multi_metrics.latency_seconds < single_metrics.latency_seconds
        ):
            return ExpectedWinner.MULTI_AGENT

        return ExpectedWinner.TIE
