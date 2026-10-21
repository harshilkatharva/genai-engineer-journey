from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from genai_core.evaluation import (
    AnswerRelevanceEvaluator,
    EvalOutput,
    EvalSample,
    ExpectedAnswer,
    FaithfulnessEvaluator,
    JudgeVerdict,
    SemanticCorrectnessEvaluator,
)
from genai_core.models import LLMError, LLMManagerResponse
from genai_core.services import LLMService


@pytest.mark.asyncio
async def test_faithfulness_evaluator_success() -> None:
    mock_llm = AsyncMock(spec=LLMService)
    mock_llm.complete_structure.return_value = LLMManagerResponse(
        structured_response=JudgeVerdict(
            score=1.0,
            passed=True,
            reasoning="All factual claims are directly supported by the context.",
        ),
        model="mock-judge-model",
    )

    evaluator = FaithfulnessEvaluator(llm_service=mock_llm)
    sample = EvalSample(id=1, query="When was the Eiffel Tower built?")
    output = EvalOutput(
        answer="The Eiffel Tower was built in 1889 for the World's Fair.",
        retrieved_contexts=[
            "The Eiffel Tower was inaugurated in 1889 for the Exposition Universelle."
        ],
    )

    result = await evaluator.evaluate(sample, output)
    assert result.score == 1.0
    assert result.passed is True
    assert "All factual claims" in (result.reason or "")
    assert mock_llm.complete_structure.called


@pytest.mark.asyncio
async def test_faithfulness_evaluator_missing_context() -> None:
    mock_llm = AsyncMock(spec=LLMService)
    evaluator = FaithfulnessEvaluator(llm_service=mock_llm)
    sample = EvalSample(id=2, query="Query")
    output = EvalOutput(answer="Some answer", retrieved_contexts=[])

    result = await evaluator.evaluate(sample, output)
    assert result.score == 0.0
    assert result.passed is False
    assert "No retrieved or sample context" in (result.reason or "")
    assert not mock_llm.complete_structure.called


@pytest.mark.asyncio
async def test_answer_relevance_evaluator_success() -> None:
    mock_llm = AsyncMock(spec=LLMService)
    mock_llm.complete_structure.return_value = LLMManagerResponse(
        structured_response=JudgeVerdict(
            score=0.9,
            passed=True,
            reasoning="Answer directly responds to the prompt with minimal fluff.",
        )
    )

    evaluator = AnswerRelevanceEvaluator(llm_service=mock_llm)
    sample = EvalSample(id=3, query="What is python?")
    output = EvalOutput(answer="Python is a high-level programming language.")

    result = await evaluator.evaluate(sample, output)
    assert result.score == 0.9
    assert result.passed is True


@pytest.mark.asyncio
async def test_semantic_correctness_evaluator_llm_failure() -> None:
    mock_llm = AsyncMock(spec=LLMService)
    mock_llm.complete_structure.return_value = LLMManagerResponse(
        error=LLMError(provider="test", code="rate_limit", message="Rate limit exceeded")
    )

    evaluator = SemanticCorrectnessEvaluator(llm_service=mock_llm)
    sample = EvalSample(id=4, query="Q", expected_answer=ExpectedAnswer(reference="Ref"))
    output = EvalOutput(answer="Answer")

    result = await evaluator.evaluate(sample, output)
    assert result.score == 0.0
    assert result.passed is False
    assert "LLM Judge execution error" in (result.reason or "")
