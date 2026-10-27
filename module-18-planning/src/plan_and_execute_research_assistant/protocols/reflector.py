"""Protocol for plan reflection and self-critique engines (Module 19 integration)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..models import Plan, StepExecutionRecord


class ReflectionResult(BaseModel):
    """Structured result returned by a PlanReflector evaluation."""

    should_replan: bool = Field(description="Whether reflection advises replanning")
    critique: str = Field(description="Detailed reflection analysis and critique")
    suggested_improvements: list[str] = Field(
        default_factory=list, description="Specific recommendations for plan revision"
    )
    score: float = Field(default=1.0, ge=0.0, le=1.0, description="Plan execution quality score")


@runtime_checkable
class PlanReflector(Protocol):
    """Protocol for reflection and self-critique engines (Module 19 integration)."""

    async def reflect(
        self,
        plan: Plan,
        step_executions: dict[str, list[StepExecutionRecord]],
        user_id: str,
    ) -> ReflectionResult:
        """Evaluate plan execution and generate reflective critique.

        Args:
            plan: The active Plan DAG under evaluation.
            step_executions: Complete record of step execution attempts.
            user_id: User identifier.

        Returns:
            ReflectionResult with critique and replan recommendation.
        """
        ...
