"""Unit tests for FileCheckpointStore."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from parallel_research_team.exceptions import CheckpointError
from parallel_research_team.models.checkpoint import CheckpointState, RunStatus
from parallel_research_team.storage.file_store import FileCheckpointStore


@pytest.mark.asyncio
async def test_save_and_get_checkpoint(tmp_path: Path) -> None:
    store = FileCheckpointStore(base_dir=tmp_path)

    state = CheckpointState(
        run_id="run_101",
        user_id="user_test",
        topic="Vector Storage Scaling",
        status=RunStatus.EXECUTING,
        used_tokens=1200,
        used_cost_usd=0.004,
    )

    await store.save_checkpoint(state)

    loaded = await store.get_checkpoint("run_101")
    assert loaded is not None
    assert loaded.run_id == "run_101"
    assert loaded.user_id == "user_test"
    assert loaded.status == RunStatus.EXECUTING
    assert loaded.used_tokens == 1200


@pytest.mark.asyncio
async def test_get_nonexistent_checkpoint(tmp_path: Path) -> None:
    store = FileCheckpointStore(base_dir=tmp_path)
    res = await store.get_checkpoint("non_existent_id")
    assert res is None


@pytest.mark.asyncio
async def test_list_checkpoints(tmp_path: Path) -> None:
    store = FileCheckpointStore(base_dir=tmp_path)

    for i in range(3):
        state = CheckpointState(
            run_id=f"run_list_{i}",
            topic=f"Topic {i}",
            status=RunStatus.COMPLETED,
        )
        await store.save_checkpoint(state)
        await asyncio.sleep(0.01)

    listed = await store.list_checkpoints(limit=10)
    assert len(listed) == 3
    # Most recently modified comes first
    assert listed[0] == "run_list_2"


@pytest.mark.asyncio
async def test_corrupted_checkpoint_file(tmp_path: Path) -> None:
    store = FileCheckpointStore(base_dir=tmp_path)
    corrupted_path = tmp_path / "checkpoint_bad.json"
    corrupted_path.write_text("{invalid_json: true", encoding="utf-8")

    with pytest.raises(CheckpointError, match="Failed to parse checkpoint file"):
        await store.get_checkpoint("bad")


@pytest.mark.asyncio
async def test_atomic_overwrite(tmp_path: Path) -> None:
    store = FileCheckpointStore(base_dir=tmp_path)

    state_v1 = CheckpointState(
        run_id="run_overwrite",
        topic="Initial Topic",
        status=RunStatus.DECOMPOSING,
        used_tokens=100,
    )
    await store.save_checkpoint(state_v1)

    state_v2 = CheckpointState(
        run_id="run_overwrite",
        topic="Initial Topic",
        status=RunStatus.COMPLETED,
        used_tokens=600,
    )
    await store.save_checkpoint(state_v2)

    loaded = await store.get_checkpoint("run_overwrite")
    assert loaded is not None
    assert loaded.status == RunStatus.COMPLETED
    assert loaded.used_tokens == 600
