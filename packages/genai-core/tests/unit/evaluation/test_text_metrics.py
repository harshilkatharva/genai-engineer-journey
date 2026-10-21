from __future__ import annotations

import pytest

from genai_core.evaluation import (
    AnswerAccuracyMetric,
    EvalOutput,
    EvalSample,
    ExactMatchMetric,
    ExpectedAnswer,
    normalize_text,
)


def test_normalize_text_characters() -> None:
    raw = "  “Hello” world—this is ‘GenAI’…  "
    norm = normalize_text(raw)
    assert norm == "\"hello\" world-this is 'genai'..."


@pytest.mark.asyncio
async def test_answer_accuracy_required_and_any_of() -> None:
    sample = EvalSample(
        id=1,
        query="What happened to the ship?",
        expected_answer=ExpectedAnswer(
            required=["Pharaon"],
            any_of=["Captain died", "Captain passed away"],
        ),
    )
    # Output contains both required "Pharaon" and alternative "Captain died"
    output = EvalOutput(
        answer="The ship Pharaon arrived in port after the captain died of brain fever."
    )

    metric = AnswerAccuracyMetric()
    result = await metric.evaluate(sample, output)

    assert result.score == 1.0
    assert result.passed is True
    assert result.details["matched_required"] == ["Pharaon"]
    assert result.details["any_of_satisfied"] is True


@pytest.mark.asyncio
async def test_answer_accuracy_missing_required() -> None:
    sample = EvalSample(
        id=2,
        query="Details?",
        expected_answer=ExpectedAnswer(
            required=["Pharaon", "Marseilles"],
            any_of=[],
        ),
    )
    # Only mentions Pharaon, misses Marseilles
    output = EvalOutput(answer="The ship Pharaon entered the bay.")

    metric = AnswerAccuracyMetric(pass_threshold=1.0)
    result = await metric.evaluate(sample, output)

    assert result.score == 0.5
    assert result.passed is False
    assert result.details["missing_required"] == ["Marseilles"]


@pytest.mark.asyncio
async def test_exact_match_metric() -> None:
    sample = EvalSample(
        id=3,
        query="Exact test",
        expected_answer=ExpectedAnswer(reference="Château d'If"),
    )
    metric = ExactMatchMetric(normalized=True)

    # Curly quote vs straight quote normalized match
    output1 = EvalOutput(answer="Château d’If")
    res1 = await metric.evaluate(sample, output1)
    assert res1.score == 1.0
    assert res1.passed is True

    # Differing text
    output2 = EvalOutput(answer="Bastille")
    res2 = await metric.evaluate(sample, output2)
    assert res2.score == 0.0
    assert res2.passed is False
