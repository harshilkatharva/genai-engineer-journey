"""Unit tests for tool registry factory and risk tiering."""

from __future__ import annotations

import pytest
from genai_core import RiskLevel

from research_and_support_agent.memory.manager import AgentMemoryManager
from research_and_support_agent.tools.registry import create_agent_tool_registry


@pytest.mark.asyncio
async def test_tool_registry_registration(memory_manager: AgentMemoryManager) -> None:
    """Verify that all 16 tools are registered with appropriate risk classifications."""
    registry = create_agent_tool_registry(memory_manager=memory_manager)
    specs = registry.get_specs()
    assert len(specs) == 16

    # Verify risk classifications
    read_only_tools = registry.get_specs(max_risk=RiskLevel.READ_ONLY)
    assert len(read_only_tools) == 6

    reversible_tools = registry.get_specs(max_risk=RiskLevel.REVERSIBLE_WRITE)
    assert len(reversible_tools) == 9  # 6 read-only + 3 reversible

    all_tools = registry.get_specs(max_risk=RiskLevel.IRREVERSIBLE)
    assert len(all_tools) == 16  # 6 read-only + 3 reversible + 7 irreversible
