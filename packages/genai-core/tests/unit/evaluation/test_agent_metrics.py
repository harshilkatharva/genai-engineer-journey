from __future__ import annotations

import pytest
from pydantic import BaseModel

from genai_core.evaluation import (
    EvalOutput,
    EvalSample,
    JSONComplianceMetric,
    ToolSelectionMetric,
)


class UserProfileSchema(BaseModel):
    name: str
    age: int


@pytest.mark.asyncio
async def test_tool_selection_metric() -> None:
    sample = EvalSample(
        id=1,
        query="Book flight and hotel",
        expected_tools=["search_flights", "book_hotel"],
    )

    # Correct tools executed
    output1 = EvalOutput(tool_calls=[{"name": "search_flights"}, {"name": "book_hotel"}])
    metric = ToolSelectionMetric()
    res1 = await metric.evaluate(sample, output1)
    assert res1.score == 1.0
    assert res1.passed is True

    # Partial execution
    output2 = EvalOutput(tool_calls=[{"name": "search_flights"}])
    res2 = await metric.evaluate(sample, output2)
    assert res2.score == 0.5
    assert res2.passed is False


@pytest.mark.asyncio
async def test_json_compliance_metric_plain_and_markdown() -> None:
    metric = JSONComplianceMetric()
    sample = EvalSample(id=2, query="Extract JSON")

    # Plain JSON
    out1 = EvalOutput(answer='{"key": "value", "count": 10}')
    res1 = await metric.evaluate(sample, out1)
    assert res1.score == 1.0
    assert res1.passed is True

    # Markdown fenced JSON
    out2 = EvalOutput(answer='```json\n{"key": "value"}\n```')
    res2 = await metric.evaluate(sample, out2)
    assert res2.score == 1.0
    assert res2.passed is True

    # Malformed JSON
    out3 = EvalOutput(answer="This is not JSON: {unclosed")
    res3 = await metric.evaluate(sample, out3)
    assert res3.score == 0.0
    assert res3.passed is False


@pytest.mark.asyncio
async def test_json_compliance_schema_validation() -> None:
    metric = JSONComplianceMetric(target_schema=UserProfileSchema)
    sample = EvalSample(id=3, query="Extract User")

    # Valid schema adherence
    out_valid = EvalOutput(answer='{"name": "Alice", "age": 30}')
    res_valid = await metric.evaluate(sample, out_valid)
    assert res_valid.score == 1.0
    assert res_valid.passed is True

    # Schema violation (missing age, age wrong type)
    out_invalid = EvalOutput(answer='{"name": "Alice", "age": "thirty"}')
    res_invalid = await metric.evaluate(sample, out_invalid)
    assert res_invalid.score == 0.5
    assert res_invalid.passed is False
