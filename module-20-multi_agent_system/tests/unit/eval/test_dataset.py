"""Unit tests for the 15-question benchmark dataset."""

from __future__ import annotations

from parallel_research_team.eval.dataset import (
    BENCHMARK_15_QUESTIONS,
    get_15_benchmark_questions,
    get_benchmark_question_by_id,
)
from parallel_research_team.eval.models import EvaluationCategory, ExpectedWinner


def test_benchmark_dataset_contains_15_questions() -> None:
    questions = get_15_benchmark_questions()
    assert len(questions) == 15
    assert len(BENCHMARK_15_QUESTIONS) == 15


def test_benchmark_dataset_unique_ids() -> None:
    questions = get_15_benchmark_questions()
    ids = [q.id for q in questions]
    assert len(ids) == len(set(ids)), "Duplicate question IDs detected in benchmark dataset."


def test_benchmark_dataset_all_categories_covered() -> None:
    questions = get_15_benchmark_questions()
    categories = {q.category for q in questions}
    assert EvaluationCategory.SIMPLE_FACTUAL in categories
    assert EvaluationCategory.LINEAR_SYNTHESIS in categories
    assert EvaluationCategory.MULTI_FACETED_COMPARISON in categories
    assert EvaluationCategory.ARCHITECTURAL_SURVEY in categories


def test_benchmark_dataset_expected_winner_distribution() -> None:
    questions = get_15_benchmark_questions()
    single_wins = [q for q in questions if q.expected_winner == ExpectedWinner.SINGLE_AGENT]
    multi_wins = [q for q in questions if q.expected_winner == ExpectedWinner.MULTI_AGENT]

    # Categories 1 and 2 (7 questions) should expect single agent to win on cost/speed
    assert len(single_wins) == 7
    # Categories 3 and 4 (8 questions) should expect multi-agent to win on quality/depth
    assert len(multi_wins) == 8


def test_get_benchmark_question_by_id() -> None:
    q01 = get_benchmark_question_by_id("q01_clarissa_recipient")
    assert q01 is not None
    assert q01.id == "q01_clarissa_recipient"
    assert "Clarissa" in q01.query
    assert "Anna Howe" in q01.ground_truth_reference

    missing = get_benchmark_question_by_id("non_existent_id")
    assert missing is None
