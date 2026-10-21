"""Benchmark evaluation metrics and aggregations for Module 17."""

from __future__ import annotations

from genai_core.evaluation.metrics.base import Metric
from genai_core.models.evaluation_model import EvalOutput, EvalSample, MetricResult


class StepBudgetMetric(Metric):
    """Evaluates whether the agent completed the task within reasonable step limits."""

    def __init__(self, max_allowed_steps: int = 10, name: str = "step_budget") -> None:
        self.name = name
        self.max_allowed_steps = max_allowed_steps

    async def evaluate(self, sample: EvalSample, output: EvalOutput) -> MetricResult:
        steps = output.token_usage.get("steps", 1) if output.token_usage else 1
        passed = steps <= self.max_allowed_steps
        return MetricResult(
            metric_name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            reason=f"Executed {steps} steps (limit: {self.max_allowed_steps})",
            details={"steps": steps, "limit": self.max_allowed_steps},
        )


class CostBudgetMetric(Metric):
    """Evaluates whether the agent stayed within the monetary cost budget."""

    def __init__(self, max_allowed_cost: float = 0.05, name: str = "cost_budget") -> None:
        self.name = name
        self.max_allowed_cost = max_allowed_cost

    async def evaluate(self, sample: EvalSample, output: EvalOutput) -> MetricResult:
        cost = (
            output.raw_response.get("cost_usd", 0.0)
            if isinstance(output.raw_response, dict)
            else 0.0
        )
        passed = cost <= self.max_allowed_cost
        return MetricResult(
            metric_name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            reason=f"Incurred ${cost:.6f} USD (budget: ${self.max_allowed_cost:.6f})",
            details={"cost_usd": cost, "budget": self.max_allowed_cost},
        )


class KeywordComplianceMetric(Metric):
    """Verifies that mandatory ground-truth keywords are present in the final answer."""

    def __init__(self, name: str = "keyword_compliance") -> None:
        self.name = name

    async def evaluate(self, sample: EvalSample, output: EvalOutput) -> MetricResult:
        if not sample.expected_answer or not sample.expected_answer.required:
            return MetricResult(
                metric_name="keyword_compliance",
                score=1.0,
                passed=True,
                reason="No mandatory keywords specified.",
            )

        text = (output.answer or "").lower()
        required_words = [w.lower() for w in sample.expected_answer.required]
        found = [w for w in required_words if w in text]
        score = len(found) / len(required_words)
        passed = score == 1.0

        return MetricResult(
            metric_name="keyword_compliance",
            score=score,
            passed=passed,
            reason=f"Matched {len(found)}/{len(required_words)} required terms",
            details={"matched": found, "missing": [w for w in required_words if w not in found]},
        )
