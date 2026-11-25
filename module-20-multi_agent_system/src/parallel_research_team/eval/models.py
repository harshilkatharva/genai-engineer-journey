"""Data models for single-agent vs multi-agent evaluation pipeline."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EvaluationCategory(StrEnum):
    """Evaluation question archetypes."""

    SIMPLE_FACTUAL = "simple_factual"
    LINEAR_SYNTHESIS = "linear_synthesis"
    MULTI_FACETED_COMPARISON = "multi_faceted_comparison"
    ARCHITECTURAL_SURVEY = "architectural_survey"


class ExpectedWinner(StrEnum):
    """Anticipated or evaluated superior system."""

    SINGLE_AGENT = "single_agent"
    MULTI_AGENT = "multi_agent"
    TIE = "tie"


class EvaluationQuestion(BaseModel):
    """Specification of an evaluation benchmark question."""

    id: str = Field(description="Unique question identifier, e.g. 'q01_clarissa_recipient'")
    query: str = Field(description="The exact user research question or prompt")
    category: EvaluationCategory = Field(description="Benchmark archetype")
    expected_winner: ExpectedWinner = Field(description="Predicted superior paradigm")
    single_agent_hypothesis: str = Field(description="Why single agent should or shouldn't help")
    multi_agent_hypothesis: str = Field(description="Why multi-agent should or shouldn't help")
    ground_truth_reference: str = Field(
        description="Key factual elements and ground-truth reference points"
    )
    rubric_focus: list[str] = Field(
        default_factory=list,
        description="Target evaluation dimensions (e.g. ['factuality', 'coverage'])",
    )


class AgentExecutionMetrics(BaseModel):
    """Performance and resource consumption metrics for an agent execution."""

    system_name: str = Field(description="'single_agent' or 'multi_agent'")
    status: str = Field(description="Execution lifecycle status")
    tokens: int = Field(default=0, ge=0, description="Total tokens consumed")
    cost_usd: float = Field(default=0.0, ge=0.0, description="Total cost in USD")
    latency_seconds: float = Field(default=0.0, ge=0.0, description="Elapsed wall-clock seconds")
    tool_calls_count: int = Field(default=0, ge=0, description="Total tool invocations made")
    sources_count: int = Field(default=0, ge=0, description="Distinct sources gathered or cited")
    error_message: str | None = Field(default=None, description="Error message if failed")


class QualityScores(BaseModel):
    """LLM-as-a-judge quality evaluation scores on a 1.0 to 5.0 scale."""

    factuality: float = Field(ge=1.0, le=5.0, description="Evidence grounding and truthfulness")
    comprehensiveness: float = Field(ge=1.0, le=5.0, description="Depth and scope coverage")
    clarity: float = Field(ge=1.0, le=5.0, description="Structure, formatting, and readability")
    relevance: float = Field(
        ge=1.0, le=5.0, description="Direct answering without extraneous noise"
    )
    overall_score: float = Field(ge=1.0, le=5.0, description="Weighted average quality score")
    reasoning: str = Field(default="", description="Judge evaluation explanation")


class QuestionEvaluationResult(BaseModel):
    """Comprehensive comparative evaluation result for a single question."""

    question_id: str
    query: str
    category: EvaluationCategory
    expected_winner: ExpectedWinner

    single_agent_answer: str = Field(default="")
    single_agent_metrics: AgentExecutionMetrics
    single_agent_quality: QualityScores

    multi_agent_answer: str = Field(default="")
    multi_agent_metrics: AgentExecutionMetrics
    multi_agent_quality: QualityScores

    winner_actual: ExpectedWinner
    delta_tokens: int = Field(
        description="multi_agent_tokens - single_agent_tokens (positive means multi-agent used more)"
    )
    delta_cost_usd: float = Field(
        description="multi_agent_cost - single_agent_cost (positive means multi-agent cost more)"
    )
    delta_latency_seconds: float = Field(
        description="multi_agent_latency - single_agent_latency (positive means multi-agent was slower)"
    )
    delta_quality_score: float = Field(
        description="multi_agent_quality - single_agent_quality (positive means multi-agent scored higher)"
    )
    completed_at: str = Field(description="ISO timestamp of evaluation completion")


class EvaluationRunSummary(BaseModel):
    """Aggregated summary of an evaluation run across all evaluated benchmark questions."""

    run_id: str
    created_at: str
    total_questions: int
    single_agent_wins: int = 0
    multi_agent_wins: int = 0
    ties: int = 0

    avg_single_latency_s: float = 0.0
    avg_multi_latency_s: float = 0.0
    avg_single_cost_usd: float = 0.0
    avg_multi_cost_usd: float = 0.0
    avg_single_quality: float = 0.0
    avg_multi_quality: float = 0.0

    results: list[QuestionEvaluationResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
