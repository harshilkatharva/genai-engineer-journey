"""Plan and PlanStep domain models."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from .enums import StepStatus


class PlanStep(BaseModel):
    """An individual actionable step within a structured execution plan."""

    id: str = Field(description="Unique step identifier, e.g. 'step_1'")
    description: str = Field(description="Actionable instruction or objective for the step")
    tool_name_hint: str | None = Field(
        default=None, description="Suggested tool category, specialist domain, or tool name"
    )
    depends_on: list[str] | None = Field(
        default=None, description="List of predecessor step IDs required before execution"
    )
    success_criteria: str = Field(
        description="Explicit, verifiable success condition to evaluate step completion"
    )
    status: StepStatus = Field(
        default=StepStatus.PENDING, description="Current execution lifecycle status of the step"
    )
    result_summary: str | None = Field(
        default=None, description="Consolidated deliverable or outcome summary after execution"
    )
    attempts: int = Field(default=0, ge=0, description="Total execution attempts for this step")
    error_message: str | None = Field(
        default=None, description="Detailed error information if the step failed"
    )


class Plan(BaseModel):
    """Structured plan represented as a Directed Acyclic Graph (DAG) of steps."""

    goal: str = Field(description="Top-level research or operations goal")
    steps: list[PlanStep] = Field(description="List of plan steps forming an executable DAG")
    version: int = Field(default=1, ge=1, description="Monotonically increasing plan version")
    parent_version: int | None = Field(
        default=None, ge=1, description="Version ID of the parent plan if derived or replanned"
    )
    created_by: str = Field(
        default="planner_llm",
        description="Author entity of this plan version ('planner_llm', 'human_reviewer', 'replanner')",
    )
    assumptions: list[str] = Field(
        default_factory=list, description="Explicit assumptions made during plan generation"
    )
    change_summary: str | None = Field(
        default=None, description="Summary of changes compared to parent_version"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when this plan version was created",
    )


class PlanVersion(BaseModel):
    """Snapshot metadata for an immutable historical version of a plan."""

    version: int = Field(ge=1, description="Plan version number")
    plan: Plan = Field(description="Plan snapshot data")
    created_by: str = Field(description="Author entity who created this version")
    parent_version: int | None = Field(default=None, ge=1, description="Predecessor version number")
    change_summary: str | None = Field(default=None, description="Diff or modification notes")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of version persistence",
    )
