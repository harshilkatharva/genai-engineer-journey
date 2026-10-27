"""Unit tests for FileRunRecordStore and RunRecord persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from plan_and_execute_research_assistant.models import (
    Plan,
    PlanStep,
    RunRecord,
    RunStatus,
    StepExecutionRecord,
    StepStatus,
    StepVerificationResult,
)
from plan_and_execute_research_assistant.storage import FileRunRecordStore


@pytest.fixture
def temp_runs_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary runs directory."""
    runs_dir = tmp_path / "runs"
    return runs_dir


@pytest.fixture
def sample_run_record() -> RunRecord:
    """Fixture providing a realistic RunRecord with plan versions and step executions."""
    plan_v1 = Plan(
        goal="Provision AWS infrastructure",
        steps=[
            PlanStep(id="step_1", description="Terraform init", success_criteria="Init done"),
            PlanStep(
                id="step_2",
                description="Terraform apply",
                depends_on=["step_1"],
                success_criteria="Apply done",
            ),
        ],
        version=1,
        created_by="planner_llm",
    )
    plan_v2 = Plan(
        goal="Provision AWS infrastructure",
        steps=[
            PlanStep(id="step_1", description="Terraform init", success_criteria="Init done"),
            PlanStep(
                id="step_2",
                description="Terraform apply",
                depends_on=["step_1"],
                success_criteria="Apply done",
            ),
            PlanStep(
                id="step_3",
                description="Health check ping",
                depends_on=["step_2"],
                success_criteria="Endpoint 200 OK",
            ),
        ],
        version=2,
        parent_version=1,
        created_by="human_reviewer",
        change_summary="Added health check verification step",
    )

    step_exec = StepExecutionRecord(
        step_id="step_1",
        plan_version=1,
        status=StepStatus.COMPLETED,
        attempts=1,
        depends_on=[],
        tokens_used=1200,
        cost_usd=0.005,
        duration_ms=450.0,
        result_summary="Terraform initialized successfully.",
        verification=StepVerificationResult(
            step_id="step_1",
            passed=True,
            programmatic_passed=True,
            model_passed=True,
            score=0.95,
            reasoning="Init exit code 0",
        ),
    )

    return RunRecord(
        run_id="test_run_12345",
        goal="Provision AWS infrastructure",
        user_id="devops_lead",
        status=RunStatus.COMPLETED,
        final_deliverable="Infrastructure provisioned and verified.",
        plan_versions=[plan_v1, plan_v2],
        step_executions={"step_1": [step_exec]},
        total_tokens=1200,
        total_cost_usd=0.005,
        duration_ms=2500.0,
    )


@pytest.mark.asyncio
async def test_file_store_directory_creation(temp_runs_dir: Path) -> None:
    """Verify FileRunRecordStore creates directory if missing."""
    assert not temp_runs_dir.exists()
    store = FileRunRecordStore(base_dir=temp_runs_dir)
    assert temp_runs_dir.exists()
    assert store.base_dir == temp_runs_dir


@pytest.mark.asyncio
async def test_save_and_get_run_roundtrip(
    temp_runs_dir: Path, sample_run_record: RunRecord
) -> None:
    """Verify saving and retrieving a full RunRecord preserves all details."""
    store = FileRunRecordStore(base_dir=temp_runs_dir)

    await store.save_run(sample_run_record)

    loaded = await store.get_run("test_run_12345")
    assert loaded is not None
    assert loaded.run_id == sample_run_record.run_id
    assert loaded.goal == sample_run_record.goal
    assert loaded.user_id == sample_run_record.user_id
    assert loaded.status == RunStatus.COMPLETED
    assert loaded.final_deliverable == "Infrastructure provisioned and verified."
    assert len(loaded.plan_versions) == 2
    assert loaded.plan_versions[0].version == 1
    assert loaded.plan_versions[1].version == 2
    assert loaded.plan_versions[1].created_by == "human_reviewer"
    assert "step_1" in loaded.step_executions
    assert loaded.step_executions["step_1"][0].status == StepStatus.COMPLETED
    assert loaded.total_tokens == 1200
    assert loaded.total_cost_usd == 0.005


@pytest.mark.asyncio
async def test_get_nonexistent_run_returns_none(temp_runs_dir: Path) -> None:
    """Verify querying a non-existent run ID returns None without raising."""
    store = FileRunRecordStore(base_dir=temp_runs_dir)
    result = await store.get_run("non_existent_run_999")
    assert result is None


@pytest.mark.asyncio
async def test_list_runs_filtering_and_limits(temp_runs_dir: Path) -> None:
    """Verify list_runs filters by user_id and respects limit."""
    store = FileRunRecordStore(base_dir=temp_runs_dir)

    run_alice_1 = RunRecord(
        run_id="run_a1", goal="Goal 1", user_id="alice", status=RunStatus.COMPLETED
    )
    run_alice_2 = RunRecord(
        run_id="run_a2", goal="Goal 2", user_id="alice", status=RunStatus.FAILED
    )
    run_bob_1 = RunRecord(run_id="run_b1", goal="Goal 3", user_id="bob", status=RunStatus.COMPLETED)

    await store.save_run(run_alice_1)
    await store.save_run(run_alice_2)
    await store.save_run(run_bob_1)

    # All runs
    all_runs = await store.list_runs()
    assert len(all_runs) == 3

    # Filtered by user alice
    alice_runs = await store.list_runs(user_id="alice")
    assert len(alice_runs) == 2
    assert all(r.user_id == "alice" for r in alice_runs)

    # Filtered by user bob
    bob_runs = await store.list_runs(user_id="bob")
    assert len(bob_runs) == 1
    assert bob_runs[0].run_id == "run_b1"

    # Limit
    limited = await store.list_runs(limit=2)
    assert len(limited) == 2


@pytest.mark.asyncio
async def test_atomic_overwrite(temp_runs_dir: Path, sample_run_record: RunRecord) -> None:
    """Verify saving updated record cleanly updates file on disk."""
    store = FileRunRecordStore(base_dir=temp_runs_dir)

    sample_run_record.status = RunStatus.EXECUTING
    await store.save_run(sample_run_record)

    loaded_1 = await store.get_run(sample_run_record.run_id)
    assert loaded_1 is not None
    assert loaded_1.status == RunStatus.EXECUTING

    # Update to COMPLETED
    sample_run_record.status = RunStatus.COMPLETED
    sample_run_record.final_deliverable = "Done!"
    await store.save_run(sample_run_record)

    loaded_2 = await store.get_run(sample_run_record.run_id)
    assert loaded_2 is not None
    assert loaded_2.status == RunStatus.COMPLETED
    assert loaded_2.final_deliverable == "Done!"


@pytest.mark.asyncio
async def test_corrupted_run_file_handled_gracefully(temp_runs_dir: Path) -> None:
    """Verify invalid JSON file is skipped in list_runs and returns None in get_run."""
    store = FileRunRecordStore(base_dir=temp_runs_dir)

    # Save a valid record
    valid_rec = RunRecord(run_id="valid_1", goal="Valid goal", user_id="alice")
    await store.save_run(valid_rec)

    # Write corrupt file directly
    corrupt_file = temp_runs_dir / "run_corrupt_bad.json"
    corrupt_file.write_text("{ this is NOT valid JSON }", encoding="utf-8")

    # get_run on corrupt file should safely return None
    assert await store.get_run("corrupt_bad") is None

    # list_runs should return the 1 valid record without raising
    runs = await store.list_runs()
    assert len(runs) == 1
    assert runs[0].run_id == "valid_1"
