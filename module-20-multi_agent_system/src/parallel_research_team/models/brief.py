"""Domain models for research briefs and decomposition effort levels."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BriefType(StrEnum):
    """Categorization of research brief objectives."""

    FACTUAL_RETRIEVAL = "factual_retrieval"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    EXPLORATORY_DEEP_DIVE = "exploratory_deep_dive"
    DIRECT_SYNTHESIS = "direct_synthesis"


class EffortLevel(StrEnum):
    """Decomposition strategy determined by prompt complexity."""

    DIRECT = "direct"
    PARALLEL_DECOMPOSED = "parallel_decomposed"


class OutputFormatSpec(BaseModel):
    """JSON-serializable specification describing expected worker deliverable structure."""

    model_config = ConfigDict(frozen=True)

    schema_name: str = Field(
        default="WorkerResearchOutput",
        description="Target Pydantic model name for the output",
    )
    required_fields: list[str] = Field(
        default_factory=lambda: ["findings_summary", "claims", "sources"],
        description="Mandatory JSON fields expected in the output",
    )
    guidelines: str = Field(
        default="Structured findings with claims and source citations",
        description="Format guidelines provided to the worker prompt",
    )


class WorkerBrief(BaseModel):
    """Typed research brief assigned to an independent research worker."""

    brief_id: str = Field(description="Unique brief identifier, e.g. 'brief_01'")
    title: str = Field(description="Short descriptive title of the assigned sub-topic")
    topic: str = Field(description="Overarching research topic")
    objective: str = Field(description="Specific research question or hypothesis to investigate")
    brief_type: BriefType = Field(default=BriefType.EXPLORATORY_DEEP_DIVE)
    worker_type: str = Field(
        default="default",
        description="Worker archetype name dispatched via WorkerRegistry factory",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Optional list of predecessor brief IDs required before this brief can run",
    )
    task_boundaries: list[str] = Field(
        description="Explicit boundaries: what is in-scope vs out-of-scope to prevent overlap"
    )
    expected_output_format: OutputFormatSpec = Field(
        default_factory=OutputFormatSpec,
        description="Typed, serializable schema reference for the worker deliverable",
    )
    suggested_tools: list[str] = Field(
        default_factory=lambda: ["search_web", "search_knowledge_base"],
        description="Tools recommended for this brief",
    )
    max_steps: int = Field(default=5, ge=1, le=10, description="Per-worker step ceiling")
    timeout_seconds: float = Field(
        default=45.0, ge=1.0, le=180.0, description="Per-worker timeout in seconds"
    )

    @field_validator("brief_id", "title", "topic", "objective")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Field cannot be empty or pure whitespace")
        return trimmed

    @field_validator("task_boundaries")
    @classmethod
    def validate_boundaries(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("task_boundaries must contain at least one boundary specification")
        return v
