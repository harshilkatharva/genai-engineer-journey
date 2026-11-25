"""Unit tests for WorkerOutputValidator."""

from __future__ import annotations

from parallel_research_team.models import (
    OutputFormatSpec,
    WorkerBrief,
    WorkerExecutionResult,
    WorkerResearchOutput,
    WorkerStatus,
)
from parallel_research_team.scheduling.validator import WorkerOutputValidator


def test_validator_valid_completed_result() -> None:
    validator = WorkerOutputValidator()
    brief = WorkerBrief(
        brief_id="b1",
        title="Title",
        topic="Topic",
        objective="Objective",
        task_boundaries=["In scope", "Out of scope"],
    )
    output = WorkerResearchOutput(
        brief_id="b1",
        findings_summary="Comprehensive research findings on agent systems.",
        sources=[{"source_id": "src_1", "chunk_id": "chk_1", "content": "Text"}],
        claims=[
            {
                "claim_id": "c1",
                "text": "Agents execute tasks.",
                "citations": [{"source_id": "src_1", "chunk_id": "chk_1"}],
            }
        ],
    )
    result = WorkerExecutionResult(
        brief_id="b1",
        status=WorkerStatus.COMPLETED,
        output=output,
    )

    validated = validator.validate(result, brief)
    assert validated.status == WorkerStatus.COMPLETED
    assert validated.output is not None


def test_validator_none_output_fails() -> None:
    validator = WorkerOutputValidator()
    brief = WorkerBrief(
        brief_id="b2",
        title="Title",
        topic="Topic",
        objective="Objective",
        task_boundaries=["In", "Out"],
    )
    result = WorkerExecutionResult(
        brief_id="b2",
        status=WorkerStatus.COMPLETED,
        output=None,
    )

    validated = validator.validate(result, brief)
    assert validated.status == WorkerStatus.FAILED
    assert "completed without output deliverable" in (validated.error_message or "")


def test_validator_empty_findings_summary_downgrades() -> None:
    validator = WorkerOutputValidator()
    brief = WorkerBrief(
        brief_id="b3",
        title="Title",
        topic="Topic",
        objective="Objective",
        task_boundaries=["In", "Out"],
    )
    # Bypass pydantic whitespace validator using construct / model_validate
    output = WorkerResearchOutput.model_construct(
        brief_id="b3",
        findings_summary="",
        sources=[],
        claims=[],
        limitations=[],
    )
    result = WorkerExecutionResult(
        brief_id="b3",
        status=WorkerStatus.COMPLETED,
        output=output,
    )

    validated = validator.validate(result, brief)
    assert validated.status == WorkerStatus.PARTIAL
    assert "empty findings summary" in (validated.error_message or "")


def test_validator_phantom_citations_flagged() -> None:
    validator = WorkerOutputValidator()
    brief = WorkerBrief(
        brief_id="b4",
        title="Title",
        topic="Topic",
        objective="Objective",
        task_boundaries=["In", "Out"],
    )
    # Output has a claim citing (src_phantom, chk_phantom) but sources list is empty
    output = WorkerResearchOutput(
        brief_id="b4",
        findings_summary="Findings with fake citation.",
        sources=[],
        claims=[
            {
                "claim_id": "c1",
                "text": "Claim text.",
                "citations": [{"source_id": "src_phantom", "chunk_id": "chk_phantom"}],
            }
        ],
    )
    result = WorkerExecutionResult(
        brief_id="b4",
        status=WorkerStatus.COMPLETED,
        output=output,
    )

    validated = validator.validate(result, brief)
    assert validated.status == WorkerStatus.PARTIAL
    assert validated.output is not None
    assert any("reference sources not present" in lim for lim in validated.output.limitations)


def test_validator_required_format_fields_checked() -> None:
    validator = WorkerOutputValidator()
    brief = WorkerBrief(
        brief_id="b5",
        title="Title",
        topic="Topic",
        objective="Objective",
        task_boundaries=["In", "Out"],
        expected_output_format=OutputFormatSpec(
            schema_name="CustomOutput",
            required_fields=["Methodology", "Performance"],
        ),
    )
    output = WorkerResearchOutput(
        brief_id="b5",
        findings_summary="This summary discusses Methodology but leaves out the other.",
        sources=[],
        claims=[],
    )
    result = WorkerExecutionResult(
        brief_id="b5",
        status=WorkerStatus.COMPLETED,
        output=output,
    )

    validated = validator.validate(result, brief)
    assert validated.output is not None
    assert any("Performance" in lim for lim in validated.output.limitations)


def test_validator_preserves_failed_or_timeout_result() -> None:
    validator = WorkerOutputValidator()
    brief = WorkerBrief(
        brief_id="b6",
        title="Title",
        topic="Topic",
        objective="Objective",
        task_boundaries=["In", "Out"],
    )
    result = WorkerExecutionResult(
        brief_id="b6",
        status=WorkerStatus.TIMEOUT,
        error_message="Worker timed out after 45s.",
    )

    validated = validator.validate(result, brief)
    assert validated.status == WorkerStatus.TIMEOUT
    assert validated.error_message == "Worker timed out after 45s."


def test_validator_field_names_not_required_in_summary_prose() -> None:
    validator = WorkerOutputValidator()
    brief = WorkerBrief(
        brief_id="b7",
        title="Title",
        topic="Topic",
        objective="Objective",
        task_boundaries=["In", "Out"],
    )
    output = WorkerResearchOutput(
        brief_id="b7",
        findings_summary="Quantum annealing demonstrates quadratic speedups over simulated annealing.",
        sources=[{"source_id": "src_1", "chunk_id": "chk_1", "content": "Text"}],
        claims=[
            {
                "claim_id": "c1",
                "text": "Speedups occur in specific regimes.",
                "citations": [{"source_id": "src_1", "chunk_id": "chk_1"}],
            }
        ],
    )
    result = WorkerExecutionResult(
        brief_id="b7",
        status=WorkerStatus.COMPLETED,
        output=output,
    )

    validated = validator.validate(result, brief)
    assert validated.status == WorkerStatus.COMPLETED
    assert validated.output is not None
    # Must NOT have appended spurious "Output format missing expected elements"
    assert not any("missing expected elements" in lim for lim in validated.output.limitations)


def test_validator_phantom_citations_discard_claims_and_downgrade_with_sources() -> None:
    validator = WorkerOutputValidator()
    brief = WorkerBrief(
        brief_id="b8",
        title="Title",
        topic="Topic",
        objective="Objective",
        task_boundaries=["In", "Out"],
    )
    output = WorkerResearchOutput(
        brief_id="b8",
        findings_summary="Valid findings text.",
        sources=[{"source_id": "src_real", "chunk_id": "chk_real", "content": "Real evidence."}],
        claims=[
            {
                "claim_id": "c_real",
                "text": "Real claim grounded in evidence.",
                "citations": [{"source_id": "src_real", "chunk_id": "chk_real"}],
            },
            {
                "claim_id": "c_phantom",
                "text": "Hallucinated claim with no evidence.",
                "citations": [{"source_id": "src_fake", "chunk_id": "chk_fake"}],
            },
        ],
    )
    result = WorkerExecutionResult(
        brief_id="b8",
        status=WorkerStatus.COMPLETED,
        output=output,
    )

    validated = validator.validate(result, brief)
    # Must downgrade to PARTIAL because of phantom citations
    assert validated.status == WorkerStatus.PARTIAL
    assert validated.output is not None
    # Phantom claim must be discarded, real claim preserved
    assert len(validated.output.claims) == 1
    assert validated.output.claims[0]["claim_id"] == "c_real"
    assert any("reference sources not present" in lim for lim in validated.output.limitations)
