from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from memory_system.models import (
    ActionCheck,
    ActionEpisode,
    ConversationMessage,
    LongTermMemoryRecord,
    MemoryCandidate,
    MemorySearchResult,
    PinnedGoal,
    WorkingContext,
    utc_now,
)


# Verifies memory data models validate and serialize their values.
def test_memory_models_validate_and_dump() -> None:
    message = ConversationMessage(role="user", content="hello")
    goal = PinnedGoal(content="Finish the task")
    memory = LongTermMemoryRecord(content="Prefers tea", user_id="user-1")

    assert message.model_dump()["role"] == "user"
    assert goal.model_dump()["content"] == "Finish the task"
    assert memory.model_dump()["category"] == "fact"
    with pytest.raises(ValidationError):
        ConversationMessage.model_validate({"role": "invalid", "content": "hello"})


def test_memory_context_and_search_models_validate_invariants() -> None:
    record = LongTermMemoryRecord(content="Prefers tea", user_id="user-1")
    result = MemorySearchResult(memory=record, similarity=0.9, rank_score=0.8)
    context = WorkingContext(
        pinned_goals=[PinnedGoal(content="Keep it short")],
        messages=[ConversationMessage(role="user", content="hello")],
        token_count=4,
        token_budget=8,
    )

    assert result.superseded is False
    assert context.token_count <= context.token_budget
    with pytest.raises(ValidationError):
        WorkingContext(pinned_goals=[], messages=[], token_count=-1, token_budget=0)
    with pytest.raises(ValidationError):
        PinnedGoal(content="")


def test_action_episode_rejects_extra_fields_and_invalid_status() -> None:
    now = utc_now()
    values = {
        "user_id": "user-1",
        "task_id": "task-1",
        "action_type": "send",
        "target": "team",
        "occurred_at": now,
        "expires_at": now,
    }

    with pytest.raises(ValidationError):
        ActionEpisode.model_validate({**values, "unexpected": "value"})
    with pytest.raises(ValidationError):
        ActionEpisode.model_validate({**values, "status": "unknown"})
    check = ActionCheck(already_attempted=False, recommendation="execute")
    assert check.episode is None
    with pytest.raises(ValidationError):
        ActionCheck.model_validate({"already_attempted": False, "recommendation": "repeat"})
    assert utc_now().tzinfo == UTC

    naive = datetime.fromisoformat("2030-01-01")
    with pytest.raises(ValidationError, match="timezone"):
        MemoryCandidate(content="temporary preference", expires_at=naive)
    with pytest.raises(ValidationError, match="timezone"):
        LongTermMemoryRecord(content="fact", user_id="user-1", created_at=naive)
    with pytest.raises(ValidationError, match="timezone"):
        ActionEpisode(
            user_id="user-1",
            task_id="task-1",
            action_type="send",
            target="recipient",
            occurred_at=naive,
            expires_at=naive,
        )
