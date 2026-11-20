"""Unit tests for ParallelResearchTeam orchestrator facade."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from parallel_research_team.core.config import ResearchTeamConfig
from parallel_research_team.exceptions import BudgetExceededError
from parallel_research_team.models import (
    BriefType,
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
        brief_coverages={"brief_01": "Fully analyzed"},
    )


async def test_orchestrator_empty_topic_raises(test_config: ResearchTeamConfig) -> None:
    team = ParallelResearchTeam(config=test_config)
    with pytest.raises(ValueError, match="cannot be empty"):
        await team.run(topic="   ")


async def test_orchestrator_end_to_end_direct_run(
    test_config: ResearchTeamConfig, tmp_path: Path
) -> None:
    store = FileCheckpointStore(base_dir=tmp_path / "checkpoints", config=test_config)

    mock_decomp = AsyncMock()
    brief = WorkerBrief(
        brief_id="brief_01",
        title="Direct Search",
        topic="What is HNSW?",
        objective="Look up HNSW",
        brief_type=BriefType.FACTUAL_RETRIEVAL,
        task_boundaries=["Focus on graph indexing"],
        expected_output_format=OutputFormatSpec(),
    )
    mock_decomp.decompose.return_value = ResearchDecomposition(
        topic="What is HNSW?",
        effort_level=EffortLevel.DIRECT,
        worker_count=1,
        worker_briefs=[brief],
        rationale="Narrow query",
    )

    mock_scheduler = AsyncMock()
    source = SourceReference(
        source_id="src_01",
        chunk_id="chk_01",
        title="HNSW Docs",
        content="Evidence content text.",
    )
    worker_res = WorkerExecutionResult(
        brief_id="brief_01",
        status=WorkerStatus.COMPLETED,
        output=WorkerResearchOutput(
            brief_id="brief_01",
            findings_summary="HNSW is a graph index",
            claims=[{"claim_id": "cl_1", "text": "HNSW is fast"}],
            sources=[source.model_dump()],
        ),
        tokens_used=800,
        cost_usd=0.002,
        duration_ms=450.0,
    )
    mock_scheduler.schedule_and_execute.return_value = {"brief_01": worker_res}

    mock_synthesis = AsyncMock()
    mock_synthesis.synthesize.return_value = _create_mock_report("What is HNSW?")

    mock_verifier = AsyncMock()
    mock_verifier.verify.return_value = SynthesisVerificationResult(
        passed=True,
        gate1_passed=True,
        gate2_passed=True,
        summary="All gates passed",
    )

    team = ParallelResearchTeam(
        config=test_config,
        decomposition_engine=mock_decomp,
        scheduler=mock_scheduler,
        synthesis_engine=mock_synthesis,
        verifier=mock_verifier,
        checkpoint_store=store,
    )

    result = await team.run(topic="What is HNSW?")

    assert result.status == RunStatus.COMPLETED
    assert result.report is not None
    assert result.report.title == "Synthesized Test Report"
    assert result.decomposition is not None
    assert result.decomposition.effort_level == EffortLevel.DIRECT
    assert result.metrics.completed_workers == 1
    assert result.metrics.total_tokens > 0

    saved_checkpoint = await store.get_checkpoint(result.run_id)
    assert saved_checkpoint is not None
    assert saved_checkpoint.status == RunStatus.COMPLETED


async def test_orchestrator_parallel_run_with_worker_failure_isolation(
    test_config: ResearchTeamConfig, tmp_path: Path
) -> None:
    store = FileCheckpointStore(base_dir=tmp_path / "checkpoints", config=test_config)

    mock_decomp = AsyncMock()
    b1 = WorkerBrief(
        brief_id="brief_01",
        title="Part 1",
        topic="Comparative Study",
        objective="Analyze Part 1",
        brief_type=BriefType.COMPARATIVE_ANALYSIS,
        task_boundaries=["Bound 1"],
        expected_output_format=OutputFormatSpec(),
    )
    b2 = WorkerBrief(
        brief_id="brief_02",
        title="Part 2",
        topic="Comparative Study",
        objective="Analyze Part 2",
        brief_type=BriefType.COMPARATIVE_ANALYSIS,
        task_boundaries=["Bound 2"],
        expected_output_format=OutputFormatSpec(),
    )
    mock_decomp.decompose.return_value = ResearchDecomposition(
        topic="Comparative Study",
        effort_level=EffortLevel.PARALLEL_DECOMPOSED,
        worker_count=2,
        worker_briefs=[b1, b2],
        rationale="Broad comparative study",
    )

    source = SourceReference(
        source_id="src_01",
        chunk_id="chk_01",
        title="Part 1 Docs",
        content="Evidence content text.",
    )
    r1 = WorkerExecutionResult(
        brief_id="brief_01",
        status=WorkerStatus.COMPLETED,
        output=WorkerResearchOutput(
            brief_id="brief_01",
            findings_summary="Findings for part 1",
            claims=[],
            sources=[source.model_dump()],
        ),
        tokens_used=1000,
        cost_usd=0.003,
    )
    r2 = WorkerExecutionResult(
        brief_id="brief_02",
        status=WorkerStatus.FAILED,
        error_message="External API timeout",
        tokens_used=200,
        cost_usd=0.0005,
    )

    mock_scheduler = AsyncMock()
    mock_scheduler.schedule_and_execute.return_value = {"brief_01": r1, "brief_02": r2}

    mock_synthesis = AsyncMock()
    mock_report = _create_mock_report("Comparative Study")
    mock_report.limitations.append("Brief 'brief_02' research incomplete: External API timeout")
    mock_report.unresolved_limitations.append(
        "Brief 'brief_02' research incomplete: External API timeout"
    )
    mock_synthesis.synthesize.return_value = mock_report

    mock_verifier = AsyncMock()
    mock_verifier.verify.return_value = SynthesisVerificationResult(
        passed=True, gate1_passed=True, gate2_passed=True, summary="Passed"
    )

    team = ParallelResearchTeam(
        config=test_config,
        decomposition_engine=mock_decomp,
        scheduler=mock_scheduler,
        synthesis_engine=mock_synthesis,
        verifier=mock_verifier,
        checkpoint_store=store,
    )

    result = await team.run(topic="Comparative Study")

    assert result.status == RunStatus.PARTIALLY_COMPLETED
    assert result.report is not None
    assert result.metrics.completed_workers == 1
    assert result.metrics.failed_workers == 1
    assert any("brief_02" in lim for lim in result.report.limitations)


async def test_orchestrator_budget_exceeded_handled_gracefully(
    test_config: ResearchTeamConfig, tmp_path: Path
) -> None:
    store = FileCheckpointStore(base_dir=tmp_path / "checkpoints", config=test_config)

    mock_decomp = AsyncMock()
    b1 = WorkerBrief(
        brief_id="brief_01",
        title="Part 1",
        topic="Big Study",
        objective="Obj 1",
        brief_type=BriefType.FACTUAL_RETRIEVAL,
        task_boundaries=["B1"],
        expected_output_format=OutputFormatSpec(),
    )
    mock_decomp.decompose.return_value = ResearchDecomposition(
        topic="Big Study",
        effort_level=EffortLevel.DIRECT,
        worker_count=1,
        worker_briefs=[b1],
        rationale="Decomp",
    )

    mock_scheduler = AsyncMock()
    mock_scheduler.schedule_and_execute.side_effect = BudgetExceededError(
        "Token ceiling of 10,000 exceeded"
    )

    team = ParallelResearchTeam(
        config=test_config,
        decomposition_engine=mock_decomp,
        scheduler=mock_scheduler,
        checkpoint_store=store,
    )

    result = await team.run(topic="Big Study")

    assert result.status == RunStatus.FAILED
    assert result.error_message is not None
    assert "budget" in result.error_message.lower()


async def test_orchestrator_all_workers_failed(
    test_config: ResearchTeamConfig, tmp_path: Path
) -> None:
    store = FileCheckpointStore(base_dir=tmp_path / "checkpoints", config=test_config)

    mock_decomp = AsyncMock()
    b1 = WorkerBrief(
        brief_id="brief_01",
        title="Part 1",
        topic="Study",
        objective="Obj 1",
        brief_type=BriefType.FACTUAL_RETRIEVAL,
        task_boundaries=["B1"],
        expected_output_format=OutputFormatSpec(),
    )
    mock_decomp.decompose.return_value = ResearchDecomposition(
        topic="Study",
        effort_level=EffortLevel.DIRECT,
        worker_count=1,
        worker_briefs=[b1],
        rationale="Decomp",
    )

    mock_scheduler = AsyncMock()
    r1 = WorkerExecutionResult(
        brief_id="brief_01",
        status=WorkerStatus.FAILED,
        error_message="Network unreachable",
    )
    mock_scheduler.schedule_and_execute.return_value = {"brief_01": r1}

    team = ParallelResearchTeam(
        config=test_config,
        decomposition_engine=mock_decomp,
        scheduler=mock_scheduler,
        checkpoint_store=store,
    )

    result = await team.run(topic="Study")

    assert result.status == RunStatus.FAILED
    assert result.error_message is not None
    assert "All research workers failed" in result.error_message
