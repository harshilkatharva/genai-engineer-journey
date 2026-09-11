from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RetryEvaluationCase:
    query: str
    expected_terms: tuple[str, ...]


RETRY_CASES: tuple[RetryEvaluationCase, ...] = (
    RetryEvaluationCase(
        query="What is the cancellation window and refund eligibility?",
        expected_terms=("cancellation", "refund"),
    ),
    RetryEvaluationCase(
        query="Which deployment limits apply to batch indexing?",
        expected_terms=("deployment", "batch", "indexing"),
    ),
    RetryEvaluationCase(
        query="How are failed payments retried and reported?",
        expected_terms=("payments", "retried", "reported"),
    ),
)


@dataclass(frozen=True)
class EvaluationResult:
    case: RetryEvaluationCase
    naive_score: float
    graph_score: float


def grounded_term_recall(answer: str, expected_terms: Sequence[str]) -> float:
    normalized = answer.casefold()
    return sum(term.casefold() in normalized for term in expected_terms) / len(expected_terms)


async def compare_retry_cases(
    naive_answer: Callable[[str], Awaitable[str]],
    graph_answer: Callable[[str], Awaitable[str]],
    cases: Sequence[RetryEvaluationCase] = RETRY_CASES,
) -> list[EvaluationResult]:
    """Compare Module 9 single-shot RAG with the retry graph on retry-shaped queries."""
    results = []
    for case in cases:
        naive = await naive_answer(case.query)
        graph = await graph_answer(case.query)
        results.append(
            EvaluationResult(
                case=case,
                naive_score=grounded_term_recall(naive, case.expected_terms),
                graph_score=grounded_term_recall(graph, case.expected_terms),
            )
        )
    return results
