"""Core infrastructure and shared runtime primitives."""

from __future__ import annotations

from .budget import SharedRunBudget
from .config import ResearchTeamConfig, get_research_team_config

__all__ = [
    "ResearchTeamConfig",
    "SharedRunBudget",
    "get_research_team_config",
]
