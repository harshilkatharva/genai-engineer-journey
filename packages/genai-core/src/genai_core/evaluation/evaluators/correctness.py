from __future__ import annotations

from typing import TYPE_CHECKING

from ...models.evaluation_model import (
    EvalOutput,
    EvalSample,
    JudgeVerdict,
    MetricResult,
)

if TYPE_CHECKING:
    from ...services.llm_services import LLMService


CORRECTNESS_PROMPT = """You are an impartial evaluation judge assessing the SEMANTIC CORRECTNESS of an answer against a ground-truth reference answer.

Question:
{query}

Ground-Truth Reference Answer:
{reference}

Generated Answer:
{answer}

Instructions:
1. Compare the Generated Answer to the Ground-Truth Reference Answer.
2. Focus on factual alignment and meaning rather than stylistic or wording variations.
3. If the Generated Answer accurately conveys the exact meaning and key facts of the reference answer, score 1.0.
4. If it partially agrees or misses details, score proportionally. If it directly contradicts the reference, score 0.0.
5. Set passed to true if score >= {threshold}.
6. Provide concise, objective reasoning.
"""


class SemanticCorrectnessEvaluator:
    """Evaluates semantic factual agreement between a generated answer and a reference answer."""

    def __init__(
        self,
        llm_service: LLMService,
        provider: str | None = None,
        name: str = "semantic_correctness",
        pass_threshold: float = 0.8,
    ) -> None:
        self.llm_service = llm_service
        self.provider = provider
        self.name = name
        self.pass_threshold = pass_threshold

    async def evaluate(self, sample: EvalSample, output: EvalOutput) -> MetricResult:
        ref = sample.expected_answer.reference if sample.expected_answer else None
        if not ref:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                reason="No reference answer specified to compare against.",
            )

        answer_str = (output.answer or "").strip()
        if not answer_str:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                reason="Generated answer is empty.",
            )

        prompt = CORRECTNESS_PROMPT.format(
            query=sample.query,
            reference=ref,
            answer=answer_str,
            threshold=self.pass_threshold,
        )

        resp = await self.llm_service.complete_structure(
            prompt=prompt,
            response_schema=JudgeVerdict,
            provider=self.provider,
        )

        if resp.error is not None:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                reason=f"LLM Judge execution error: {resp.error.message}",
            )

        verdict = resp.structured_response
        if isinstance(verdict, JudgeVerdict):
            return MetricResult(
                metric_name=self.name,
                score=verdict.score,
                passed=verdict.passed,
                reason=verdict.reasoning,
                details={"judge_model": resp.model, "reasoning": verdict.reasoning},
            )

        return MetricResult(
            metric_name=self.name,
            score=0.0,
            passed=False,
            reason="LLM judge failed to produce a structured JudgeVerdict.",
        )
