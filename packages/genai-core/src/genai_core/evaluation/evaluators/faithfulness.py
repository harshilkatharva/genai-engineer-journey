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


FAITHFULNESS_PROMPT = """You are an impartial evaluation judge assessing the FAITHFULNESS of an answer against provided source context.

Source Context:
{context}

Question:
{query}

Generated Answer:
{answer}

Instructions:
1. Identify all factual claims made in the Generated Answer.
2. Verify if each claim can be directly inferred from the Source Context.
3. If the answer introduces claims, facts, or details not supported by the Source Context, it is a hallucination.
4. Score 1.0 if completely grounded in the context, 0.0 if entirely ungrounded, or a fraction proportional to supported claims.
5. Set passed to true if score >= {threshold}.
6. Provide concise, objective reasoning.
"""


class FaithfulnessEvaluator:
    """Evaluates whether generated answer claims are strictly grounded in retrieved context (hallucination detection)."""

    def __init__(
        self,
        llm_service: LLMService,
        provider: str | None = None,
        name: str = "faithfulness",
        pass_threshold: float = 0.8,
    ) -> None:
        self.llm_service = llm_service
        self.provider = provider
        self.name = name
        self.pass_threshold = pass_threshold

    async def evaluate(self, sample: EvalSample, output: EvalOutput) -> MetricResult:
        contexts = output.retrieved_contexts or sample.context or []
        context_str = "\n---\n".join(contexts).strip()
        answer_str = (output.answer or "").strip()

        if not answer_str:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                reason="Generated answer is empty.",
            )

        if not context_str:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                reason="No retrieved or sample context provided to verify faithfulness.",
            )

        prompt = FAITHFULNESS_PROMPT.format(
            context=context_str,
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
