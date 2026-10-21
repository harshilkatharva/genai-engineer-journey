from __future__ import annotations

from typing import Any

from ...models.evaluation_model import EvalOutput, EvalSample, MetricResult

_PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "…": "...",
    }
)


def normalize_text(text: Any) -> str:
    """Normalize text by translating unicode punctuation, casefolding, and collapsing whitespace."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.translate(_PUNCTUATION_TRANSLATION).casefold().split())


class AnswerAccuracyMetric:
    """Evaluates answer fact accuracy using normalized required and alternative substring matches."""

    def __init__(self, name: str = "answer_accuracy", pass_threshold: float = 1.0) -> None:
        self.name = name
        self.pass_threshold = pass_threshold

    async def evaluate(self, sample: EvalSample, output: EvalOutput) -> MetricResult:
        if sample.expected_answer is None:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                reason="No expected answer specified for this sample.",
            )

        generated = normalize_text(output.answer or "")
        required_answers = [a for a in sample.expected_answer.required if a.strip()]
        any_of_answers = [a for a in sample.expected_answer.any_of if a.strip()]

        has_any_of = bool(any_of_answers)
        total_reqs = len(required_answers) + (1 if has_any_of else 0)

        if total_reqs == 0:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                reason="No required or alternative answers specified.",
            )

        matched_required = [ans for ans in required_answers if normalize_text(ans) in generated]
        missing_required = [ans for ans in required_answers if ans not in matched_required]

        matched_any_of = [ans for ans in any_of_answers if normalize_text(ans) in generated]
        any_of_satisfied = bool(matched_any_of) if has_any_of else True

        satisfied_reqs = len(matched_required) + (1 if (has_any_of and any_of_satisfied) else 0)
        accuracy = satisfied_reqs / total_reqs

        passed = accuracy >= self.pass_threshold
        reason = f"Accuracy {accuracy:.2f} ({satisfied_reqs}/{total_reqs} answer requirements met)"

        return MetricResult(
            metric_name=self.name,
            score=accuracy,
            passed=passed,
            reason=reason,
            details={
                "matched_required": matched_required,
                "missing_required": missing_required,
                "matched_any_of": matched_any_of,
                "any_of_satisfied": any_of_satisfied if has_any_of else None,
                "satisfied_requirements": satisfied_reqs,
                "total_requirements": total_reqs,
            },
        )


class ExactMatchMetric:
    """Evaluates exact matching against an expected reference answer string."""

    def __init__(
        self,
        name: str = "exact_match",
        normalized: bool = True,
    ) -> None:
        self.name = name
        self.normalized = normalized

    async def evaluate(self, sample: EvalSample, output: EvalOutput) -> MetricResult:
        if sample.expected_answer is None or not sample.expected_answer.reference:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                reason="No reference answer specified.",
            )

        ref = sample.expected_answer.reference
        gen = output.answer or ""

        if self.normalized:
            match = normalize_text(ref) == normalize_text(gen)
        else:
            match = ref.strip() == gen.strip()

        score = 1.0 if match else 0.0
        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=match,
            reason="Exact match satisfied" if match else "Output does not match reference",
            details={"reference": ref, "generated": gen},
        )
