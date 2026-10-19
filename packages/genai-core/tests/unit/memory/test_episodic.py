from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from genai_core.exceptions import MemoryValidationError
from genai_core.memory.episodic import EpisodicMemory, canonical_fingerprint
from genai_core.memory.in_memory_store import InMemoryMemoryStore
from genai_core.models.memory_model import ActionStatus


def test_canonical_fingerprint() -> None:
    # Key order invariance
    p1 = {"b": 2, "a": 1, "nested": {"y": "yes", "x": "no"}}
    p2 = {"nested": {"x": "no", "y": "yes"}, "a": 1, "b": 2}
    h1 = canonical_fingerprint("TOOL_A", p1, target="server1")
    h2 = canonical_fingerprint("tool_a", p2, target="SERVER1")
    assert h1 == h2

    # Different params produce different hash
    h3 = canonical_fingerprint("tool_a", {"a": 2}, target="server1")
    assert h1 != h3

    # Empty action_type raises error
    with pytest.raises(MemoryValidationError):
        canonical_fingerprint("", p1)

    # Non-serializable params
    with pytest.raises(MemoryValidationError):
        canonical_fingerprint("tool", {"unsupported": object()})


@pytest.mark.asyncio
async def test_episodic_memory_record_and_idempotency_check() -> None:
    store = InMemoryMemoryStore()
    em = EpisodicMemory(store=store, retention_days=14)
    user_id = "agent_runner"

    # Initially action should be cleared to execute
    check1 = await em.check(
        user_id=user_id,
        action_type="restart_service",
        parameters={"service": "redis", "port": 6379},
    )
    assert check1.status == "execute"
    assert check1.previous_episode is None

    # Record action execution
    ep = await em.record(
        user_id=user_id,
        action_type="restart_service",
        parameters={"port": 6379, "service": "redis"},
        result_summary="Redis restarted with PID 1234",
        status=ActionStatus.SUCCESS,
    )
    assert ep.parameter_hash is not None

    # Now checking again should indicate do_not_repeat
    check2 = await em.check(
        user_id=user_id,
        action_type="restart_service",
        parameters={"service": "redis", "port": 6379},
    )
    assert check2.status == "do_not_repeat"
    assert check2.previous_episode is not None
    assert check2.previous_episode.id == ep.id

    # Different parameters can execute
    check3 = await em.check(
        user_id=user_id,
        action_type="restart_service",
        parameters={"service": "postgres", "port": 5432},
    )
    assert check3.status == "execute"

    # Purge
    past = datetime.now(UTC) - timedelta(days=20)
    purged = await em.purge(older_than=past, user_id=user_id)
    assert purged == 0

    await store.close()
