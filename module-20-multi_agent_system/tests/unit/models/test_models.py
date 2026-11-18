"""Unit tests for domain models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from parallel_research_team.models.brief import (
    BriefType,
    EffortLevel,
    OutputFormatSpec,
    WorkerBrief,
)
from parallel_research_team.models.checkpoint import CheckpointState, RunStatus
from parallel_research_team.models.decomposition import ResearchDecomposition
from parallel_research_team.models.result import ExecutionMetrics, ResearchTeamRunResult
from parallel_research_team.models.worker import (
    WorkerExecutionResult,
    WorkerResearchOutput,
    WorkerStatus,
)


def test_output_format_spec() -> None:
    spec = OutputFormatSpec()
    assert spec.schema_name == "WorkerResearchOutput"
    assert "findings_summary" in spec.required_fields

    dumped = spec.model_dump()
    loaded = OutputFormatSpec.model_validate(dumped)
    assert loaded == spec


def test_worker_brief_validation(sample_brief: WorkerBrief) -> None:
    assert sample_brief.brief_id == "brief_01"
    assert sample_brief.brief_type == BriefType.COMPARATIVE_ANALYSIS
    assert len(sample_brief.task_boundaries) == 2

    # Empty objective fails
    with pytest.raises(ValidationError):
        WorkerBrief(
            brief_id="b2",
            title="Title",
            topic="Topic",
            objective="   ",
            task_boundaries=["Valid boundary"],
        )

    # Empty task_boundaries fails
    with pytest.raises(ValidationError):
        WorkerBrief(
            brief_id="b2",
            title="Title",
            topic="Topic",
            objective="Valid",
            task_boundaries=[],
        )


def test_worker_research_output() -> None:
    output = WorkerResearchOutput(
        brief_id="brief_01",
        findings_summary="HNSW indexing delivers sub-5ms query latency on 1M vectors.",
        claims=[{"claim_id": "c1", "text": "HNSW delivers sub-5ms query latency."}],
        sources=[{"source_id": "src_1", "chunk_id": "chk_1", "content": "Text"}],
    )
    assert output.brief_id == "brief_01"
    assert len(output.claims) == 1

    with pytest.raises(ValidationError):
        WorkerResearchOutput(brief_id="b1", findings_summary="")


def test_worker_execution_result() -> None:
    res = WorkerExecutionResult(
        brief_id="brief_01",
        status=WorkerStatus.COMPLETED,
        tokens_used=1500,
        cost_usd=0.005,
        duration_ms=450.0,
    )
    assert res.status == WorkerStatus.COMPLETED
    assert res.tokens_used == 1500


def test_research_decomposition(sample_brief: WorkerBrief) -> None:
    decomp = ResearchDecomposition(
        topic="Vector Indexing",
        effort_level=EffortLevel.PARALLEL_DECOMPOSED,
        rationale="Multi-faceted comparison",
        worker_count=1,
        worker_briefs=[sample_brief],
    )
    assert decomp.worker_count == 1
    assert decomp.effort_level == EffortLevel.PARALLEL_DECOMPOSED

    # Duplicate brief IDs fail
    with pytest.raises(ValidationError, match="Duplicate brief_id"):
        ResearchDecomposition(
            topic="Vector Indexing",
            effort_level=EffortLevel.PARALLEL_DECOMPOSED,
            rationale="Test",
            worker_count=2,
            worker_briefs=[sample_brief, sample_brief.model_copy(deep=True)],
        )


def test_checkpoint_state_serialization(sample_brief: WorkerBrief) -> None:
    decomp = ResearchDecomposition(
        topic="Topic",
        effort_level=EffortLevel.DIRECT,
        rationale="Simple query",
        worker_count=1,
        worker_briefs=[sample_brief],
    )
    state = CheckpointState(
        run_id="run_123",
        user_id="user_abc",
        topic="Topic",
        status=RunStatus.EXECUTING,
        decomposition=decomp,
        used_tokens=500,
        used_cost_usd=0.002,
    )

    json_str = state.model_dump_json()
    loaded = CheckpointState.model_validate_json(json_str)
    assert loaded.run_id == "run_123"
    assert loaded.status == RunStatus.EXECUTING
    assert loaded.decomposition is not None
    assert loaded.decomposition.worker_briefs[0].brief_id == "brief_01"


def test_research_team_run_result() -> None:
    metrics = ExecutionMetrics(
        total_tokens=2000,
        total_cost_usd=0.008,
        duration_ms=1200.0,
        total_workers=2,
        completed_workers=2,
    )
    result = ResearchTeamRunResult(
        run_id="run_xyz",
        status=RunStatus.COMPLETED,
        metrics=metrics,
    )
    assert result.status == RunStatus.COMPLETED
    assert result.metrics.total_tokens == 2000
