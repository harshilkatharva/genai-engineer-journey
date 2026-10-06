from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from memory_system.models import (
    ActionEpisode,
    ConversationMessage,
    LongTermMemoryRecord,
    MemoryCandidate,
    PinnedGoal,
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


# Verifies naive datetimes are rejected by memory and episode models.
def test_memory_timestamps_require_timezone() -> None:
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
