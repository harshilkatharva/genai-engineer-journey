from __future__ import annotations

from typing import Any

import pytest

from memory_system import EpisodicMemory
from memory_system.core import Settings
from tests.fakes import MemoryFakeRepository


def create_episodic() -> tuple[MemoryFakeRepository, EpisodicMemory]:
    repository = MemoryFakeRepository()
    settings = Settings(memory_embedding_dimension=2)
    return repository, EpisodicMemory(repository, settings)


@pytest.mark.asyncio
async def test_check_recommends_against_repeating_action() -> None:
    _, episodic = create_episodic()
    params: dict[str, Any] = {"recipient": "team@example.test", "subject": "Report"}
    before = await episodic.check("user-1", "task-1", "send_email", "team", params)
    assert before.recommendation == "execute"

    await episodic.record("user-1", "task-1", "send_email", "team", params, status="completed")
    after = await episodic.check(
        "user-1",
        "task-1",
        "SEND_EMAIL",
        "TEAM",
        {"subject": "Report", "recipient": "team@example.test"},
    )

    assert after.already_attempted is True
    assert after.recommendation == "do_not_repeat"
    assert after.episode is not None


def test_fingerprint_is_stable_for_mapping_order() -> None:
    left = EpisodicMemory._fingerprint("write", "record", {"a": 1, "b": 2})
    right = EpisodicMemory._fingerprint(" WRITE ", " RECORD ", {"b": 2, "a": 1})

    assert left == right
