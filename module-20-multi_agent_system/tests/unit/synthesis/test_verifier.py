"""Unit tests for ResearchVerifier integrating Gate 1 and Gate 2."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from self_improving_report_writer.models.rubric import SemanticGroundingReport

from parallel_research_team.core.config import ResearchTeamConfig
from parallel_research_team.models import (
    Citation,
    CitationType,
    Claim,
    Report,
    ReportSection,
    SourceReference,
)
from parallel_research_team.synthesis.source_pool import AuthoritativeSourcePool
from parallel_research_team.synthesis.verifier import ResearchVerifier


def _create_valid_report_and_pool() -> tuple[Report, AuthoritativeSourcePool]:
    src = SourceReference(
        source_id="src_01",
        chunk_id="chk_01",
        title="Parallel Execution Study",
        content="Parallel execution scales sub-queries across independent worker agents.",
    )
    pool = AuthoritativeSourcePool([src])

    claim = Claim(
        claim_id="claim_1",
        section_id="sec_exec",
        text="Parallel execution scales sub-queries across independent worker agents.",
        citations=[
            Citation(
                citation_id="cite_1",
                source_id="src_01",
                chunk_id="chk_01",
                citation_type=CitationType.PARAPHRASE,
            )
        ],
    )

    sec_exec = ReportSection(
        section_id="sec_exec",
        title="Executive Summary",
        content="Overview with citation [cite_1].",
        claims=[claim],
    )
    sec_body = ReportSection(
        section_id="sec_body",
        title="Body Section",
        content="Body content.",
        claims=[],
    )
    sec_concl = ReportSection(
        section_id="sec_concl",
        title="Conclusion",
        content="Conclusion content.",
        claims=[],
    )

    report = Report(
        topic="Scalability",
        title="Scaling Research",
        executive_summary=sec_exec,
        sections=[sec_body],
        conclusion=sec_concl,
    )
    return report, pool


@pytest.mark.asyncio
async def test_verifier_gate1_success_gate2_disabled(
    test_config: ResearchTeamConfig,
) -> None:
    test_config.enable_semantic_verification = False
    report, pool = _create_valid_report_and_pool()

    verifier = ResearchVerifier(config=test_config)
    result = await verifier.verify(report=report, source_pool=pool)

    assert result.passed is True
    assert result.gate1_passed is True
    assert result.gate2_passed is None
    assert result.gate1_report is not None


@pytest.mark.asyncio
async def test_verifier_gate1_failure_stops_before_gate2(
    test_config: ResearchTeamConfig,
) -> None:
    test_config.enable_semantic_verification = True
    report, _ = _create_valid_report_and_pool()
    # Empty pool means citation [cite_1] points to an unknown source!
    empty_pool = AuthoritativeSourcePool()

    verifier = ResearchVerifier(config=test_config)
    result = await verifier.verify(report=report, source_pool=empty_pool)

    assert result.passed is False
    assert result.gate1_passed is False
    assert result.gate2_passed is None
    assert len(result.issues) >= 1
    assert any("not present in application-controlled" in str(iss) for iss in result.issues)


@pytest.mark.asyncio
async def test_verifier_gate2_success_threshold(
    test_config: ResearchTeamConfig,
) -> None:
    test_config.enable_semantic_verification = True
    test_config.semantic_verification_threshold = 0.70
    report, pool = _create_valid_report_and_pool()

    mock_gate2_report = SemanticGroundingReport(
        passed=True,
        all_claims_grounded=True,
        grounding_score=0.85,
        claim_evaluations=[],
        issues=[],
        critical_count=0,
        major_count=0,
        minor_count=0,
        tokens_used=100,
        cost_usd=0.001,
    )

    verifier = ResearchVerifier(config=test_config)

    with patch(
        "parallel_research_team.synthesis.verifier.SemanticGroundingVerifier.verify_grounding",
        new_callable=AsyncMock,
    ) as mock_verify:
        mock_verify.return_value = mock_gate2_report
        result = await verifier.verify(report=report, source_pool=pool)

    assert result.passed is True
    assert result.gate1_passed is True
    assert result.gate2_passed is True
    assert "0.85" in result.summary


@pytest.mark.asyncio
async def test_verifier_gate2_failure_below_threshold(
    test_config: ResearchTeamConfig,
) -> None:
    test_config.enable_semantic_verification = True
    test_config.semantic_verification_threshold = 0.70
    report, pool = _create_valid_report_and_pool()

    mock_gate2_report = SemanticGroundingReport(
        passed=False,
        all_claims_grounded=False,
        grounding_score=0.55,
        claim_evaluations=[],
        issues=[],
        critical_count=1,
        major_count=0,
        minor_count=0,
        tokens_used=100,
        cost_usd=0.001,
    )

    verifier = ResearchVerifier(config=test_config)

    with patch(
        "parallel_research_team.synthesis.verifier.SemanticGroundingVerifier.verify_grounding",
        new_callable=AsyncMock,
    ) as mock_verify:
        mock_verify.return_value = mock_gate2_report
        result = await verifier.verify(report=report, source_pool=pool)

    assert result.passed is False
    assert result.gate1_passed is True
    assert result.gate2_passed is False
    assert "0.55" in result.summary


@pytest.mark.asyncio
async def test_verifier_gate2_failure_due_to_major_defects_above_threshold(
    test_config: ResearchTeamConfig,
) -> None:
    test_config.enable_semantic_verification = True
    test_config.semantic_verification_threshold = 0.70
    report, pool = _create_valid_report_and_pool()

    mock_gate2_report = SemanticGroundingReport(
        passed=False,
        all_claims_grounded=False,
        grounding_score=0.88,
        claim_evaluations=[],
        issues=[],
        critical_count=0,
        major_count=1,
        minor_count=0,
        tokens_used=100,
        cost_usd=0.001,
    )

    verifier = ResearchVerifier(config=test_config)

    with patch(
        "parallel_research_team.synthesis.verifier.SemanticGroundingVerifier.verify_grounding",
        new_callable=AsyncMock,
    ) as mock_verify:
        mock_verify.return_value = mock_gate2_report
        result = await verifier.verify(report=report, source_pool=pool)

    assert result.passed is False
    assert result.gate1_passed is True
    assert result.gate2_passed is False
    assert "0.88 >= threshold 0.70" in result.summary
    assert "1 major" in result.summary


@pytest.mark.asyncio
async def test_verifier_gate2_exception_shielding(
    test_config: ResearchTeamConfig,
) -> None:
    test_config.enable_semantic_verification = True
    report, pool = _create_valid_report_and_pool()

    verifier = ResearchVerifier(config=test_config)

    with patch(
        "parallel_research_team.synthesis.verifier.SemanticGroundingVerifier.verify_grounding",
        new_callable=AsyncMock,
    ) as mock_verify:
        mock_verify.side_effect = RuntimeError("Judge model rate limit")
        result = await verifier.verify(report=report, source_pool=pool)

    assert result.passed is False
    assert result.gate1_passed is True
    assert result.gate2_passed is False
    assert any("Judge model rate limit" in str(iss) for iss in result.issues)


@pytest.mark.asyncio
async def test_verifier_attaches_gate2_token_and_cost(
    test_config: ResearchTeamConfig,
) -> None:
    test_config.enable_semantic_verification = True
    report, pool = _create_valid_report_and_pool()
    verifier = ResearchVerifier(config=test_config)

    mock_gate2_report = SemanticGroundingReport(
        passed=True,
        all_claims_grounded=True,
        grounding_score=0.95,
        claim_evaluations=[],
        issues=[],
        critical_count=0,
        major_count=0,
        minor_count=0,
        tokens_used=280,
        cost_usd=0.0012,
    )

    with patch(
        "parallel_research_team.synthesis.verifier.SemanticGroundingVerifier.verify_grounding",
        new_callable=AsyncMock,
    ) as mock_verify:
        mock_verify.return_value = mock_gate2_report
        result = await verifier.verify(report=report, source_pool=pool)

    assert result.passed is True
    assert result.tokens_used == 280
    assert result.cost_usd == 0.0012
