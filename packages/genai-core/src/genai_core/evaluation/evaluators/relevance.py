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


RELEVANCE_PROMPT = """You are an impartial evaluation judge assessing the RELEVANCE of a generated answer to the user's question.

Question:
{query}

Generated Answer:
{answer}

Instructions:
1. Determine whether the answer directly and specifically addresses what was asked in the question.
2. Penalize evasiveness, unnecessary fluff, off-topic tangents, or incomplete answers.
3. If the answer is completely non-responsive, score 0.0. If it directly answers the question concisely and thoroughly, score 1.0.
4. Set passed to true if score >= {threshold}.
5. Provide concise, objective reasoning.
"""


class AnswerRelevanceEvaluator:
    """Evaluates whether the generated answer directly addresses the user query."""

    def __init__(
        self,
        llm_service: LLMService,
        provider: str | None = None,
        name: str = "answer_relevance",
        pass_threshold: float = 0.8,
    ) -> None:
        self.llm_service = llm_service
        self.provider = provider
        self.name = name
        self.pass_threshold = pass_threshold

    async def evaluate(self, sample: EvalSample, output: EvalOutput) -> MetricResult:
        answer_str = (output.answer or "").strip()
        if not answer_str:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                reason="Generated answer is empty.",
            )

        prompt = RELEVANCE_PROMPT.format(
            query=sample.query,
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
