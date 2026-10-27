"""Unit tests for PlanningMemoryManager and episodic/LTM recall."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from genai_core import (
    ActionEpisode,
    ActionStatus,
    LongTermMemoryRecord,
    MemorySearchResult,
    MemoryService,
)

from plan_and_execute_research_assistant.core import AssistantConfig
from plan_and_execute_research_assistant.memory import PlanningMemoryManager
from plan_and_execute_research_assistant.models import Plan, PlanStep


@pytest.fixture
def mock_memory_service() -> MagicMock:
    """Mock MemoryService from genai-core."""
    svc = MagicMock(spec=MemoryService)
    svc.search_longterm = AsyncMock(return_value=[])
    svc.episodic = MagicMock()
    svc.episodic.get_episodes = AsyncMock(return_value=[])
    svc.episodic.record = AsyncMock(
        return_value=ActionEpisode(
            user_id="user_1",
            action_type="plan_execution",
            parameter_hash="dummy_hash",
            parameters={"goal": "Test"},
            status=ActionStatus.SUCCESS,
        )
    )
    return svc


@pytest.mark.asyncio
async def test_empty_user_id_returns_empty(mock_memory_service: MagicMock) -> None:
    """Verify empty user_id short-circuits gracefully."""
    mgr = PlanningMemoryManager(memory_service=mock_memory_service)
    assert await mgr.get_user_preferences(user_id="", query="query") == []
    assert await mgr.get_similar_past_plans(user_id="", goal="goal") == []


@pytest.mark.asyncio
async def test_get_user_preferences_success(mock_memory_service: MagicMock) -> None:
    """Verify long-term preferences are recalled and extracted."""
    rec1 = LongTermMemoryRecord(
        id="rec1",
        user_id="user_123",
        content="User prefers PostgreSQL",
        category="preference",
    )
    rec2 = LongTermMemoryRecord(
        id="rec2",
        user_id="user_123",
        content="User requires strict input validation",
        category="preference",
    )
    mock_memory_service.search_longterm.return_value = [
        MemorySearchResult(record=rec1, similarity=0.9, recency_score=1.0, rank_score=0.95),
        MemorySearchResult(record=rec2, similarity=0.85, recency_score=1.0, rank_score=0.90),
    ]
    mgr = PlanningMemoryManager(memory_service=mock_memory_service)
    prefs = await mgr.get_user_preferences(user_id="user_123", query="build database")

    assert len(prefs) == 2
    assert "User prefers PostgreSQL" in prefs
    assert "User requires strict input validation" in prefs
    mock_memory_service.search_longterm.assert_awaited_once_with(
        user_id="user_123", query="build database", limit=5
    )


@pytest.mark.asyncio
async def test_get_user_preferences_error_handled(mock_memory_service: MagicMock) -> None:
    """Verify memory search errors fall back to empty list without raising."""
    mock_memory_service.search_longterm.side_effect = RuntimeError("Database unreachable")
    mgr = PlanningMemoryManager(memory_service=mock_memory_service)
    prefs = await mgr.get_user_preferences(user_id="user_123", query="query")
    assert prefs == []


@pytest.mark.asyncio
async def test_episodic_search_disabled_by_config(mock_memory_service: MagicMock) -> None:
    """Verify episodic past plan recall is skipped when config toggle is False."""
    config = AssistantConfig(enable_episodic_similarity_search=False)
    mgr = PlanningMemoryManager(memory_service=mock_memory_service, config=config)

    plans = await mgr.get_similar_past_plans(user_id="user_123", goal="Deploy service")
    assert plans == []
    # Verifies episodic store was never called
    mock_memory_service.episodic.get_episodes.assert_not_called()


@pytest.mark.asyncio
async def test_episodic_search_enabled_by_config(mock_memory_service: MagicMock) -> None:
    """Verify episodic past plan recall executes when config toggle is True."""
    config = AssistantConfig(enable_episodic_similarity_search=True)
    mock_memory_service.episodic.get_episodes.return_value = [
        ActionEpisode(
            user_id="user_123",
            action_type="plan_execution",
            parameter_hash="hash1",
            parameters={"goal": "Past deployment goal"},
            result_summary="Successfully provisioned ECS service",
            metadata={"step_descriptions": ["[step_1]: Build image", "[step_2]: Push to ECR"]},
            status=ActionStatus.SUCCESS,
        )
    ]
    mgr = PlanningMemoryManager(memory_service=mock_memory_service, config=config)

    plans = await mgr.get_similar_past_plans(user_id="user_123", goal="Deploy service")
    assert len(plans) == 1
    assert plans[0]["goal"] == "Past deployment goal"
    assert plans[0]["result_summary"] == "Successfully provisioned ECS service"
    assert len(plans[0]["steps"]) == 2


@pytest.mark.asyncio
async def test_record_plan_episode(mock_memory_service: MagicMock) -> None:
    """Verify recording a completed plan persists to episodic memory."""
    mgr = PlanningMemoryManager(memory_service=mock_memory_service)
    plan = Plan(
        goal="Audit support tickets",
        steps=[
            PlanStep(id="s1", description="Fetch tickets", success_criteria="Done"),
            PlanStep(
                id="s2", description="Audit tickets", depends_on=["s1"], success_criteria="Done"
            ),
        ],
        version=1,
    )
    episode = await mgr.record_plan_episode(
        user_id="user_123",
        goal="Audit support tickets",
        plan=plan,
        result_summary="All tickets audited",
    )
    assert episode.user_id == "user_1"
    mock_memory_service.episodic.record.assert_awaited_once()
