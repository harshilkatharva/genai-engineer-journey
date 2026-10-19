from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from genai_core.models.memory_model import MemoryCategory, MessageRole
from genai_core.services.memory_service import MemoryService


@pytest.mark.asyncio
async def test_memory_service_orchestration() -> None:
    mock_embed = AsyncMock()
    mock_embed.embed_query.return_value = [0.1, 0.2, 0.3]

    service = MemoryService(embedding_service=mock_embed)
    await service.initialize()

    user_id = "user_orchestrator"
    session_id = "sess_main"

    # 1. Short-term interactions
    msg = await service.add_message(
        user_id=user_id,
        session_id=session_id,
        role=MessageRole.USER,
        content="I am booking a flight to Tokyo",
    )
    assert msg.content == "I am booking a flight to Tokyo"

    goal = await service.pin_goal(
        user_id=user_id,
        session_id=session_id,
        goal="Focus on direct flights only",
    )
    assert goal.goal == "Focus on direct flights only"

    ctx = await service.get_working_context(user_id=user_id, session_id=session_id)
    assert len(ctx.messages) == 1
    assert len(ctx.pinned_goals) == 1

    # 2. Long-term interactions
    rec = await service.remember(
        user_id=user_id,
        content="User has a valid Japanese tourist visa",
        category=MemoryCategory.FACT,
    )
    assert rec.content == "User has a valid Japanese tourist visa"

    recalled = await service.recall(user_id=user_id, query="visa status")
    assert len(recalled) == 1
    assert recalled[0].record.content == rec.content

    # 3. Episodic interactions
    check_before = await service.check_action(
        user_id=user_id,
        action_type="book_flight",
        parameters={"flight_no": "NH101"},
    )
    assert check_before.status == "execute"

    await service.record_action(
        user_id=user_id,
        action_type="book_flight",
        parameters={"flight_no": "NH101"},
        result_summary="Booking confirmed #98765",
    )

    check_after = await service.check_action(
        user_id=user_id,
        action_type="book_flight",
        parameters={"flight_no": "NH101"},
    )
    assert check_after.status == "do_not_repeat"

    # 4. Maintenance
    purge_results = await service.purge_all(user_id=user_id)
    assert "expired_long_term" in purge_results
    assert "aged_episodes" in purge_results

    await service.close()
