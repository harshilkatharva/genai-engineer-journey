from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from memory_system import EpisodicMemory
from memory_system.core import Settings
from tests.fakes import MemoryFakeRepository, make_settings


# Creates episodic memory with an isolated in-memory repository.
def create_episodic() -> tuple[MemoryFakeRepository, EpisodicMemory]:
    repository = MemoryFakeRepository()
    settings = Settings(memory_embedding_dimension=2)
    return repository, EpisodicMemory(repository, settings)


# Verifies duplicate actions are flagged to avoid repeating them.
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


@pytest.mark.asyncio
async def test_record_sets_retention_and_check_tracks_parameter_changes() -> None:
    repository, episodic = create_episodic()
    episode = await episodic.record(
        "user-1", "task-1", " send ", " target ", status="failed", conversation_id="conv-1"
    )

    assert episode.action_type == " send "
    assert episode.target == " target "
    assert episode.status == "failed"
    assert episode.conversation_id == "conv-1"
    assert episode.expires_at - episode.occurred_at == timedelta(days=90)
    assert (await episodic.check("user-1", "task-1", "send", "target")).already_attempted is True
    assert (
        await episodic.check("user-1", "task-1", "send", "target", {"changed": True})
    ).recommendation == "execute"
    assert len(repository.episodes) == 1


@pytest.mark.asyncio
async def test_record_and_check_reject_missing_required_action_fields() -> None:
    episodic = EpisodicMemory(MemoryFakeRepository(), make_settings())

    invalid_actions = [
        ("", "task", "send", "target"),
        ("user", "", "send", "target"),
        ("user", "task", " ", "target"),
        ("user", "task", "send", " "),
    ]
    for values in invalid_actions:
        with pytest.raises(ValueError, match="required"):
            await episodic.record(*values)
        with pytest.raises(ValueError, match="required"):
            await episodic.check(*values)


@pytest.mark.asyncio
async def test_invalid_action_parameters_are_rejected_and_delete_all_delegates() -> None:
    repository, episodic = create_episodic()

    with pytest.raises(ValueError, match="JSON-serializable"):
        await episodic.record("user-1", "task-1", "update", "record", {"value": float("nan")})
    await episodic.record("user-1", "task-1", "update", "record")

    assert await episodic.delete_all("user-1") == 1
    assert await episodic.delete_all("missing-user") == 0
    with pytest.raises(ValueError, match="user_id"):
        await episodic.delete_all("")
    assert repository.episodes == []
