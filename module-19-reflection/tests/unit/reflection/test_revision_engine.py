"""Unit tests for RevisionEngine and revision prompt construction."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from genai_core import LLMError, LLMManagerResponse

from self_improving_report_writer.core.exceptions import ReflectionError
from self_improving_report_writer.models.critique import (
    Critique,
    CritiqueIssue,
    IssueCategory,
    IssueSeverity,
)
from self_improving_report_writer.models.report import Report, SourceReference
from self_improving_report_writer.reflection.revision_engine import (
    RevisionEngine,
    build_revision_prompt,
)


@pytest.fixture
def sample_critique() -> Critique:
    """Fixture returning a structured Critique with prioritized issues."""
    issue1 = CritiqueIssue(
        issue_id="CRITIQUE-001",
        category=IssueCategory.CITATION_INTEGRITY,
        severity=IssueSeverity.CRITICAL,
        affected_section_id="sec_findings",
        affected_claim_id="claim_find_1",
        explanation="Quotation passage was not found in cited source chunk.",
        suggested_correction="Use exact verbatim quote or reclassify as paraphrase.",
    )
    issue2 = CritiqueIssue(
        issue_id="CRITIQUE-002",
        category=IssueCategory.COMPLETENESS,
        severity=IssueSeverity.MAJOR,
        affected_section_id="sec_findings",
        explanation="Discussion of NISQ hardware limits is missing.",
        suggested_correction="Add context on NISQ error rates.",
    )
    return Critique(
        round_number=1,
        issues=[issue1, issue2],
        summary="Revision required for quotation and completeness defects.",
        should_revise=True,
    )


def test_build_revision_prompt(
    valid_report: Report,
    sample_sources: list[SourceReference],
    sample_critique: Critique,
) -> None:
    """build_revision_prompt should include topic, prioritized critique issues, and sources."""
    prompt = build_revision_prompt(
        current_report=valid_report,
        sources=sample_sources,
        critique=sample_critique,
    )
    assert valid_report.topic in prompt
    assert "[CRITICAL] (citation_integrity)" in prompt
    assert "Quotation passage was not found in cited source chunk." in prompt
    assert "[MAJOR] (completeness)" in prompt
    assert "untrusted_source_context" in prompt
    assert "chunk_q01" in prompt
    assert f"Increment version to {valid_report.version + 1}" in prompt


@pytest.fixture
def mock_llm_service() -> AsyncMock:
    """Mock LLMService."""
    service = AsyncMock()
    service.complete_structure = AsyncMock()
    return service


async def test_revise_report_success(
    mock_llm_service: AsyncMock,
    valid_report: Report,
    sample_sources: list[SourceReference],
    sample_critique: Critique,
) -> None:
    """RevisionEngine should generate revised report, increment version, and record provenance."""
    # Create revised clone
    revised_report = valid_report.model_copy(deep=True)
    revised_report.version = 1  # generator might repeat version 1

    mock_llm_service.complete_structure.return_value = LLMManagerResponse(
        structured_response=revised_report,
        usage={"input_tokens": 1200, "output_tokens": 800, "total_tokens": 2000},
        raw_response={"cost": 0.005},
    )

    engine = RevisionEngine(llm_service=mock_llm_service)
    result_report, tokens, cost = await engine.revise_report(
        current_report=valid_report,
        sources=sample_sources,
        critique=sample_critique,
    )

    assert result_report.version == 2
    assert result_report.created_by == "revision_llm"
    assert tokens == 2000
    assert cost == 0.005
    mock_llm_service.complete_structure.assert_awaited_once()


async def test_revise_report_dict_response(
    mock_llm_service: AsyncMock,
    valid_report: Report,
    sample_sources: list[SourceReference],
    sample_critique: Critique,
) -> None:
    """RevisionEngine should deserialize dict structured responses."""
    rep_dict = valid_report.model_dump(mode="json")
    rep_dict["version"] = 3

    mock_llm_service.complete_structure.return_value = LLMManagerResponse(
        structured_response=rep_dict,
        usage={"total_tokens": 1500},
        raw_response={"cost": 0.003},
    )

    engine = RevisionEngine(llm_service=mock_llm_service)
    result_report, tokens, cost = await engine.revise_report(
        current_report=valid_report,
        sources=sample_sources,
        critique=sample_critique,
    )

    assert result_report.version == 3
    assert result_report.created_by == "revision_llm"
    assert tokens == 1500
    assert cost == 0.003


async def test_revise_report_llm_error_raises_reflection_error(
    mock_llm_service: AsyncMock,
    valid_report: Report,
    sample_sources: list[SourceReference],
    sample_critique: Critique,
) -> None:
    """RevisionEngine should raise ReflectionError when LLM returns an error."""
    mock_llm_service.complete_structure.return_value = LLMManagerResponse(
        error=LLMError(provider="test", code="500", message="Rate limit exceeded"),
    )

    engine = RevisionEngine(llm_service=mock_llm_service)
    with pytest.raises(ReflectionError, match="Rate limit exceeded"):
        await engine.revise_report(
            current_report=valid_report,
            sources=sample_sources,
            critique=sample_critique,
        )


async def test_revise_report_exception_raises_reflection_error(
    mock_llm_service: AsyncMock,
    valid_report: Report,
    sample_sources: list[SourceReference],
    sample_critique: Critique,
) -> None:
    """RevisionEngine should wrap unexpected exceptions into ReflectionError."""
    mock_llm_service.complete_structure.side_effect = RuntimeError("Network partition")

    engine = RevisionEngine(llm_service=mock_llm_service)
    with pytest.raises(ReflectionError, match="Network partition"):
        await engine.revise_report(
            current_report=valid_report,
            sources=sample_sources,
            critique=sample_critique,
        )
