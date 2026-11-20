"""Unit tests for research team checkpoint resumption."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

from parallel_research_team.core.config import ResearchTeamConfig
from parallel_research_team.models import (
    BriefCoverage,
    BriefType,
    CheckpointState,
    Citation,
    CitationType,
    Claim,
    EffortLevel,
    OutputFormatSpec,
    Report,
    ReportSection,
    ResearchDecomposition,
    RunStatus,
    SourceReference,
    WorkerBrief,
    WorkerExecutionResult,
    WorkerResearchOutput,
    WorkerStatus,
)
from parallel_research_team.orchestrator import ParallelResearchTeam
from parallel_research_team.storage.file_store import FileCheckpointStore
from parallel_research_team.synthesis.verifier import SynthesisVerificationResult


def _create_mock_report(topic: str) -> Report:
    claim = Claim(
        claim_id="cl_1",
        section_id="sec_1",
        text="Evidence content text.",
        citations=[
            Citation(
                citation_id="cite_1",
                source_id="src_01",
                chunk_id="chk_01",
                citation_type=CitationType.PARAPHRASE,
            )
        ],
    )
    sec_exec = ReportSection(section_id="sec_exec", title="Exec", content="Exec text", claims=[])
    sec_body = ReportSection(section_id="sec_1", title="Body", content="Body text", claims=[claim])
    sec_conc = ReportSection(section_id="sec_conc", title="Conc", content="Conc text", claims=[])

    return Report(
        topic=topic,
        title="Synthesized Test Report",
        executive_summary=sec_exec,
        sections=[sec_body],
        conclusion=sec_conc,
        limitations=[],
        brief_coverages=[
            BriefCoverage(brief_id="brief_01", coverage_summary="Covered"),
            BriefCoverage(brief_id="brief_02", coverage_summary="Covered"),
        ],
    )


async def test_resume_not_found(test_config: ResearchTeamConfig, tmp_path: Path) -> None:
    store = FileCheckpointStore(base_dir=tmp_path / "checkpoints", config=test_config)
    team = ParallelResearchTeam(config=test_config, checkpoint_store=store)

    result = await team.resume("run_nonexistent_999")
    assert result.status == RunStatus.FAILED
    assert "not found" in (result.error_message or "").lower()


async def test_resume_already_completed(test_config: ResearchTeamConfig, tmp_path: Path) -> None:
    store = FileCheckpointStore(base_dir=tmp_path / "checkpoints", config=test_config)
    mock_report = _create_mock_report("Topic Completed")

    checkpoint = CheckpointState(
        run_id="run_completed_001",
        user_id="user_1",
        topic="Topic Completed",
        status=RunStatus.COMPLETED,
        final_report=mock_report.model_dump(),
        used_tokens=4500,
        used_cost_usd=0.012,
        elapsed_seconds=15.0,
    )
    await store.save_checkpoint(checkpoint)

    mock_scheduler = AsyncMock()
    mock_synthesis = AsyncMock()

    team = ParallelResearchTeam(
        config=test_config,
        checkpoint_store=store,
        scheduler=mock_scheduler,
        synthesis_engine=mock_synthesis,
    )

    result = await team.resume("run_completed_001")
    assert result.status == RunStatus.COMPLETED
    assert result.report is not None
    assert result.report.topic == "Topic Completed"
    mock_scheduler.schedule_and_execute.assert_not_called()
    mock_synthesis.synthesize.assert_not_called()


async def test_resume_executes_only_pending_briefs(
    test_config: ResearchTeamConfig, tmp_path: Path
) -> None:
    store = FileCheckpointStore(base_dir=tmp_path / "checkpoints", config=test_config)

    b1 = WorkerBrief(
        brief_id="brief_01",
        title="Subtopic 1",
        topic="Big Topic",
        objective="Analyze subtopic 1",
        brief_type=BriefType.FACTUAL_RETRIEVAL,
        task_boundaries=["B1"],
        expected_output_format=OutputFormatSpec(),
    )
    b2 = WorkerBrief(
        brief_id="brief_02",
        title="Subtopic 2",
        topic="Big Topic",
        objective="Analyze subtopic 2",
        brief_type=BriefType.FACTUAL_RETRIEVAL,
        task_boundaries=["B2"],
        expected_output_format=OutputFormatSpec(),
    )
    decomposition = ResearchDecomposition(
        topic="Big Topic",
        effort_level=EffortLevel.PARALLEL_DECOMPOSED,
        worker_count=2,
        worker_briefs=[b1, b2],
        rationale="Decomp",
    )

    source1 = SourceReference(
        source_id="src_01",
        chunk_id="chk_01",
        title="Source 1",
        content="Evidence 1",
    )
    res1 = WorkerExecutionResult(
        brief_id="brief_01",
        status=WorkerStatus.COMPLETED,
        output=WorkerResearchOutput(
            brief_id="brief_01",
            findings_summary="Findings for 1",
            claims=[],
            sources=[source1.model_dump()],
        ),
        tokens_used=1200,
        cost_usd=0.003,
    )

    checkpoint = CheckpointState(
        run_id="run_interrupted_002",
        user_id="user_2",
        topic="Big Topic",
        status=RunStatus.EXECUTING,
        effort_level=EffortLevel.PARALLEL_DECOMPOSED,
        decomposition=decomposition,
        worker_results={"brief_01": res1},
        authoritative_sources={"chk_01": source1.model_dump()},
        used_tokens=2000,
        used_cost_usd=0.006,
        elapsed_seconds=5.0,
    )
    await store.save_checkpoint(checkpoint)

    source2 = SourceReference(
        source_id="src_02",
        chunk_id="chk_02",
        title="Source 2",
        content="Evidence 2",
    )
    res2 = WorkerExecutionResult(
        brief_id="brief_02",
        status=WorkerStatus.COMPLETED,
        output=WorkerResearchOutput(
            brief_id="brief_02",
            findings_summary="Findings for 2",
            claims=[],
            sources=[source2.model_dump()],
        ),
        tokens_used=1500,
        cost_usd=0.004,
    )

    mock_scheduler = AsyncMock()
    mock_scheduler.schedule_and_execute.return_value = {"brief_02": res2}

    mock_synthesis = AsyncMock()
    mock_report = _create_mock_report("Big Topic")
    mock_synthesis.synthesize.return_value = mock_report

    mock_verifier = AsyncMock()
    mock_verifier.verify.return_value = SynthesisVerificationResult(
        passed=True, gate1_passed=True, gate2_passed=True, summary="Clean"
    )

    team = ParallelResearchTeam(
        config=test_config,
        checkpoint_store=store,
        scheduler=mock_scheduler,
        synthesis_engine=mock_synthesis,
        verifier=mock_verifier,
    )

    result = await team.resume("run_interrupted_002")

    assert result.status == RunStatus.COMPLETED
    assert result.report is not None

    mock_scheduler.schedule_and_execute.assert_called_once()
    scheduled_briefs = mock_scheduler.schedule_and_execute.call_args[1]["briefs"]
    assert len(scheduled_briefs) == 1
    assert scheduled_briefs[0].brief_id == "brief_02"

    mock_synthesis.synthesize.assert_called_once()
    synth_worker_results = mock_synthesis.synthesize.call_args[1]["worker_results"]
    assert "brief_01" in synth_worker_results
    assert "brief_02" in synth_worker_results

    updated_cp = await store.get_checkpoint("run_interrupted_002")
    assert updated_cp is not None
    assert updated_cp.status == RunStatus.COMPLETED
    assert "brief_01" in updated_cp.worker_results
    assert "brief_02" in updated_cp.worker_results
    assert "chk_01" in updated_cp.authoritative_sources
    assert "chk_02" in updated_cp.authoritative_sources
