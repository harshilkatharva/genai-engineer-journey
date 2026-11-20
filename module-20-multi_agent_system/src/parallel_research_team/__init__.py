"""Parallel Research Team multi-agent research package."""

from __future__ import annotations

from .core.config import ResearchTeamConfig, get_research_team_config
from .models import (
    CheckpointState,
    EffortLevel,
    ExecutionMetrics,
    Report,
    ResearchDecomposition,
    ResearchTeamRunResult,
    RunStatus,
    WorkerBrief,
)
from .orchestrator import ParallelResearchTeam

__version__ = "0.1.0"

__all__ = [
    "CheckpointState",
    "EffortLevel",
    "ExecutionMetrics",
    "ParallelResearchTeam",
    "Report",
    "ResearchDecomposition",
    "ResearchTeamConfig",
    "ResearchTeamRunResult",
    "RunStatus",
    "WorkerBrief",
    "get_research_team_config",
]
