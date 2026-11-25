"""Unit tests for judge scoring and models."""

from __future__ import annotations

from parallel_research_team.eval.judge import EvaluationJudge
from parallel_research_team.eval.models import (
    AgentExecutionMetrics,
    EvaluationCategory,
    ExpectedWinner,
    QualityScores,
    QuestionEvaluationResult,
)


def test_determine_winner_significant_quality_lead() -> None:
    judge = EvaluationJudge()

    single_metrics = AgentExecutionMetrics(
        system_name="single_agent",
        status="completed",
        tokens=1000,
        cost_usd=0.001,
        latency_seconds=5.0,
    )
    single_quality = QualityScores(
        factuality=3.0,
        comprehensiveness=3.0,
        clarity=3.0,
        relevance=3.0,
        overall_score=3.0,
    )

    multi_metrics = AgentExecutionMetrics(
        system_name="multi_agent",
        status="completed",
        tokens=15000,
        cost_usd=0.015,
        latency_seconds=25.0,
    )
    multi_quality = QualityScores(
        factuality=4.8,
        comprehensiveness=4.9,
        clarity=4.5,
        relevance=4.5,
        overall_score=4.75,
    )

    # Multi-agent has huge quality lead (+1.75) -> multi_agent wins
    winner = judge.determine_winner(single_metrics, single_quality, multi_metrics, multi_quality)
    assert winner == ExpectedWinner.MULTI_AGENT


def test_determine_winner_tied_quality_efficiency_wins() -> None:
    judge = EvaluationJudge()

    # Both achieve 4.5 quality (tied within 0.3)
    single_metrics = AgentExecutionMetrics(
        system_name="single_agent",
        status="completed",
        tokens=800,
        cost_usd=0.0008,
        latency_seconds=3.5,
    )
    single_quality = QualityScores(
        factuality=4.5,
        comprehensiveness=4.5,
        clarity=4.5,
        relevance=4.5,
        overall_score=4.5,
    )

    multi_metrics = AgentExecutionMetrics(
        system_name="multi_agent",
        status="completed",
        tokens=6000,
        cost_usd=0.006,
        latency_seconds=20.0,
    )
    multi_quality = QualityScores(
        factuality=4.6,
        comprehensiveness=4.5,
        clarity=4.5,
        relevance=4.5,
        overall_score=4.55,
    )

    # Quality is tied, but single agent was 6x faster and 7x cheaper -> single_agent wins
    winner = judge.determine_winner(single_metrics, single_quality, multi_metrics, multi_quality)
    assert winner == ExpectedWinner.SINGLE_AGENT


def test_question_evaluation_result_serialization() -> None:
    result = QuestionEvaluationResult(
        question_id="q01_test",
        query="test query",
        category=EvaluationCategory.SIMPLE_FACTUAL,
        expected_winner=ExpectedWinner.SINGLE_AGENT,
        single_agent_answer="answer single",
        single_agent_metrics=AgentExecutionMetrics(
            system_name="single_agent",
            status="completed",
            tokens=500,
            cost_usd=0.0005,
            latency_seconds=2.0,
        ),
        single_agent_quality=QualityScores(
            factuality=5.0,
            comprehensiveness=5.0,
            clarity=5.0,
            relevance=5.0,
            overall_score=5.0,
        ),
        multi_agent_answer="answer multi",
        multi_agent_metrics=AgentExecutionMetrics(
            system_name="multi_agent",
            status="completed",
            tokens=4000,
            cost_usd=0.004,
            latency_seconds=15.0,
        ),
        multi_agent_quality=QualityScores(
            factuality=5.0,
            comprehensiveness=5.0,
            clarity=5.0,
            relevance=5.0,
            overall_score=5.0,
        ),
        winner_actual=ExpectedWinner.SINGLE_AGENT,
        delta_tokens=3500,
        delta_cost_usd=0.0035,
        delta_latency_seconds=13.0,
        delta_quality_score=0.0,
        completed_at="2026-11-25T15:00:00Z",
    )

    dumped = result.model_dump()
    assert dumped["question_id"] == "q01_test"
    assert dumped["winner_actual"] == "single_agent"
