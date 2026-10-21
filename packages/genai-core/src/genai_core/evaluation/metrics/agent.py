from __future__ import annotations

import json

from pydantic import BaseModel, ValidationError

from ...models.evaluation_model import EvalOutput, EvalSample, MetricResult


class ToolSelectionMetric:
    """Evaluates whether expected tools were selected during execution."""

    def __init__(self, name: str = "tool_selection", match_order: bool = False) -> None:
        self.name = name
        self.match_order = match_order

    async def evaluate(self, sample: EvalSample, output: EvalOutput) -> MetricResult:
        if not sample.expected_tools:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                reason="No expected tools specified.",
            )

        executed_tools = [
            call.get("name") or call.get("tool_name")
            for call in (output.tool_calls or [])
            if isinstance(call, dict) and (call.get("name") or call.get("tool_name"))
        ]

        expected_set = set(sample.expected_tools)
        executed_set = set(executed_tools)

        matched = expected_set & executed_set
        score = len(matched) / len(expected_set)

        if self.match_order:
            order_match = executed_tools[: len(sample.expected_tools)] == sample.expected_tools
            passed = order_match and score == 1.0
            reason = "Tools matched in exact order" if passed else "Tool sequence mismatch"
        else:
            passed = score == 1.0
            reason = (
                f"Tool selection score: {score:.2f} ({len(matched)}/{len(expected_set)} matched)"
            )

        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=passed,
            reason=reason,
            details={
                "expected": sample.expected_tools,
                "executed": executed_tools,
                "missing": sorted(expected_set - executed_set),
                "unexpected": sorted(executed_set - expected_set),
            },
        )


class JSONComplianceMetric:
    """Evaluates whether model output is valid JSON and optionally adheres to a Pydantic schema."""

    def __init__(
        self,
        name: str = "json_compliance",
        target_schema: type[BaseModel] | None = None,
    ) -> None:
        self.name = name
        self.target_schema = target_schema

    async def evaluate(self, sample: EvalSample, output: EvalOutput) -> MetricResult:
        text = (output.answer or "").strip()
        if not text:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                reason="Output text is empty.",
            )

        # Strip optional markdown code fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            parsed_json = json.loads(text)
        except (json.JSONDecodeError, ValueError) as e:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                reason=f"Invalid JSON: {e}",
            )

        if self.target_schema is not None:
            try:
                self.target_schema.model_validate(parsed_json)
                return MetricResult(
                    metric_name=self.name,
                    score=1.0,
                    passed=True,
                    reason="Output conforms to target Pydantic schema.",
                )
            except ValidationError as ve:
                return MetricResult(
                    metric_name=self.name,
                    score=0.5,
                    passed=False,
                    reason=f"Valid JSON but failed schema validation: {ve.error_count()} errors",
                    details={"validation_errors": ve.errors()},
                )

        return MetricResult(
            metric_name=self.name,
            score=1.0,
            passed=True,
            reason="Output is valid JSON.",
        )
