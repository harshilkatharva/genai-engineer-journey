"""Unit tests for the self-correcting reflection engine and orchestrator integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from self_improving_report_writer.models.rubric import ClaimGroundingResult, SemanticGroundingReport

from parallel_research_team.core.config import ResearchTeamConfig
from parallel_research_team.models import (
    Citation,
    CitationType,
    Claim,
    Report,
    ReportSection,
    SourceReference,
)
from parallel_research_team.synthesis.reflection import ReflectionEngine
from parallel_research_team.synthesis.source_pool import AuthoritativeSourcePool
from parallel_research_team.synthesis.verifier import ResearchVerifier, SynthesisVerificationResult


def _create_sample_report_and_pool() -> tuple[Report, AuthoritativeSourcePool]:
    src = SourceReference(
        source_id="src_01",
        chunk_id="chk_01",
        title="Agent Performance Study",
        content="Single agents achieve lower latency on atomic queries.",
    )
    pool = AuthoritativeSourcePool([src])

    claim = Claim(
        claim_id="claim_1",
        section_id="sec_1",
        text="Single agents achieve lower latency on atomic queries.",
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
        content="Summary overview.",
        claims=[],
    )

    sec_body = ReportSection(
        section_id="sec_1",
        title="Latency Analysis",
        content="Single agents achieve lower latency on atomic queries [cite_1].",
        claims=[claim],
    )

    sec_concl = ReportSection(
        section_id="sec_concl",
        title="Conclusion",
        content="Conclusion text.",
        claims=[],
    )

    report = Report(
        topic="Agent Latency Analysis",
        title="Comparative Latency Study",
        executive_summary=sec_exec,
        sections=[sec_body],
        conclusion=sec_concl,
        limitations=[],
    )
    return report, pool


@pytest.mark.asyncio
async def test_reflection_skipped_when_verification_passed() -> None:
    config = ResearchTeamConfig(enable_reflection=True)
    engine = ReflectionEngine(config=config)
    report, pool = _create_sample_report_and_pool()

    passed_verif = SynthesisVerificationResult(
        passed=True,
        gate1_passed=True,
        gate2_passed=True,
        summary="All gates passed",
    )
    mock_verifier = MagicMock(spec=ResearchVerifier)

    outcome = await engine.reflect_and_revise(
        topic=report.topic,
        current_report=report,
        source_pool=pool,
        verification_result=passed_verif,
        verifier=mock_verifier,
    )

    assert outcome.report is report
    assert outcome.steps_executed == 0
    assert outcome.tokens_used == 0
    assert not outcome.improved
    assert not outcome.exhausted_budget


@pytest.mark.asyncio
async def test_reflection_skipped_when_disabled() -> None:
    config = ResearchTeamConfig(enable_reflection=False)
    engine = ReflectionEngine(config=config)
    report, pool = _create_sample_report_and_pool()

    failed_verif = SynthesisVerificationResult(
        passed=False,
        gate1_passed=True,
        gate2_passed=False,
        summary="Gate 2 failed",
    )
    mock_verifier = MagicMock(spec=ResearchVerifier)

    outcome = await engine.reflect_and_revise(
        topic=report.topic,
        current_report=report,
        source_pool=pool,
        verification_result=failed_verif,
        verifier=mock_verifier,
    )

    assert outcome.report is report
    assert outcome.steps_executed == 0
    assert not outcome.improved


@pytest.mark.asyncio
async def test_reflection_success_resolves_issues() -> None:
    config = ResearchTeamConfig(
        enable_reflection=True,
        max_reflection_steps=2,
        max_reflection_tokens=15_000,
        max_reflection_cost_usd=0.05,
    )
    report, pool = _create_sample_report_and_pool()

    g2_failed_report = SemanticGroundingReport(
        passed=False,
        all_claims_grounded=False,
        grounding_score=0.50,
        claim_evaluations=[
            ClaimGroundingResult(
                claim_id="claim_1",
                section_id="sec_1",
                entailment="NEUTRAL",
                score=0.50,
                reasoning="Claim text generalizes beyond source passage.",
                cited_chunk_ids=["chk_01"],
            )
        ],
    )
    initial_verif = SynthesisVerificationResult(
        passed=False,
        gate1_passed=True,
        gate2_passed=False,
        gate2_report=g2_failed_report,
        summary="Gate 2 semantic verification failed",
    )

    # Revised report candidate
    revised_report, _ = _create_sample_report_and_pool()
    revised_report.sections[0].claims[
        0
    ].text = "Single agents achieve lower latency specifically on atomic queries."

    mock_llm_response = MagicMock()
    mock_llm_response.error = None
    mock_llm_response.structured_response = revised_report
    mock_llm_response.prompt_tokens = 500
    mock_llm_response.completion_tokens = 300

    mock_llm_service = MagicMock()
    mock_llm_service.complete_structure = AsyncMock(return_value=mock_llm_response)

    engine = ReflectionEngine(config=config, llm_service=mock_llm_service)

    # Verifier will pass the revised report
    passed_verif = SynthesisVerificationResult(
        passed=True,
        gate1_passed=True,
        gate2_passed=True,
        summary="All verification gates passed",
        tokens_used=100,
        cost_usd=0.0001,
    )
    mock_verifier = MagicMock(spec=ResearchVerifier)
    mock_verifier.verify = AsyncMock(return_value=passed_verif)

    outcome = await engine.reflect_and_revise(
        topic=report.topic,
        current_report=report,
        source_pool=pool,
        verification_result=initial_verif,
        verifier=mock_verifier,
    )

    assert outcome.improved is True
    assert outcome.verification_result.passed is True
    assert outcome.steps_executed == 1
    assert outcome.tokens_used == 900  # 500 + 300 LLM + 100 verifier
    assert outcome.cost_usd > 0.0
    assert not outcome.exhausted_budget


@pytest.mark.asyncio
async def test_reflection_budget_exhaustion() -> None:
    # Set token envelope to minimum valid 1000
    config = ResearchTeamConfig(
        enable_reflection=True,
        max_reflection_steps=3,
        max_reflection_tokens=1_000,
        max_reflection_cost_usd=0.05,
    )
    report, pool = _create_sample_report_and_pool()

    g2_failed_report = SemanticGroundingReport(
        passed=False,
        all_claims_grounded=False,
        grounding_score=0.40,
        claim_evaluations=[
            ClaimGroundingResult(
                claim_id="claim_1",
                section_id="sec_1",
                entailment="CONTRADICTS",
                score=0.0,
                reasoning="Not supported.",
                cited_chunk_ids=["chk_01"],
            )
        ],
    )
    initial_verif = SynthesisVerificationResult(
        passed=False,
        gate1_passed=True,
        gate2_passed=False,
        gate2_report=g2_failed_report,
        summary="Gate 2 failed",
    )

    # Return 1100 tokens from LLM which exceeds the 1000 token envelope on step 1
    mock_llm_response = MagicMock()
    mock_llm_response.error = None
    mock_llm_response.structured_response = report
    mock_llm_response.prompt_tokens = 700
    mock_llm_response.completion_tokens = 400

    mock_llm_service = MagicMock()
    mock_llm_service.complete_structure = AsyncMock(return_value=mock_llm_response)

    still_failed_verif = SynthesisVerificationResult(
        passed=False,
        gate1_passed=True,
        gate2_passed=False,
        gate2_report=g2_failed_report,
        summary="Gate 2 failed again",
        tokens_used=50,
        cost_usd=0.0001,
    )
    mock_verifier = MagicMock(spec=ResearchVerifier)
    mock_verifier.verify = AsyncMock(return_value=still_failed_verif)

    engine = ReflectionEngine(config=config, llm_service=mock_llm_service)

    outcome = await engine.reflect_and_revise(
        topic=report.topic,
        current_report=report,
        source_pool=pool,
        verification_result=initial_verif,
        verifier=mock_verifier,
    )

    assert outcome.exhausted_budget is True
    assert outcome.steps_executed == 1
    assert outcome.verification_result.passed is False
    # Check that limitation advisory was appended
    assert any("Reflection attempted" in lim for lim in outcome.report.limitations)


@pytest.mark.asyncio
async def test_reflection_llm_exception_shielding() -> None:
    config = ResearchTeamConfig(enable_reflection=True, max_reflection_steps=2)
    report, pool = _create_sample_report_and_pool()

    initial_verif = SynthesisVerificationResult(
        passed=False,
        gate1_passed=True,
        gate2_passed=False,
        issues=[{"error": "Gate 2 defect"}],
        summary="Gate 2 failed",
    )

    mock_llm_service = MagicMock()
    mock_llm_service.complete_structure = AsyncMock(side_effect=RuntimeError("API quota exceeded"))

    mock_verifier = MagicMock(spec=ResearchVerifier)
    engine = ReflectionEngine(config=config, llm_service=mock_llm_service)

    outcome = await engine.reflect_and_revise(
        topic=report.topic,
        current_report=report,
        source_pool=pool,
        verification_result=initial_verif,
        verifier=mock_verifier,
    )

    # Does not crash, returns original report safely
    assert outcome.report is report
    assert outcome.steps_executed == 1
    assert outcome.improved is False
