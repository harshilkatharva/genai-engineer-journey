"""Unit tests for Lead Agent SynthesisEngine."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from parallel_research_team.core.budget import SharedRunBudget
from parallel_research_team.core.config import ResearchTeamConfig
from parallel_research_team.exceptions import SynthesisError
from parallel_research_team.models import (
    Citation,
    CitationType,
    Claim,
    Report,
    ReportSection,
    SourceReference,
    WorkerExecutionResult,
    WorkerResearchOutput,
    WorkerStatus,
)
from parallel_research_team.prompts.manager import PromptTemplateManager
from parallel_research_team.synthesis.engine import SynthesisEngine
from parallel_research_team.synthesis.source_pool import AuthoritativeSourcePool


def _create_mock_report() -> Report:
    sec_exec = ReportSection(
        section_id="sec_exec",
        title="Executive Summary",
        content="Overview of multi-agent concurrency findings [cite_1].",
        claims=[
            Claim(
                claim_id="claim_1",
                section_id="sec_exec",
                text="Multi-agent systems improve execution throughput.",
                citations=[
                    Citation(
                        citation_id="cite_1",
                        source_id="src_1",
                        chunk_id="chk_1",
                        citation_type=CitationType.PARAPHRASE,
                    )
                ],
            )
        ],
    )
    sec_body = ReportSection(
        section_id="sec_findings",
        title="Key Findings",
        content="Detailed findings on locking mechanisms.",
        claims=[],
    )
    sec_concl = ReportSection(
        section_id="sec_conclusion",
        title="Conclusion",
        content="Synthesis conclusion.",
        claims=[],
    )
    return Report(
        topic="Multi-Agent Concurrency",
        title="Synthesis of Multi-Agent Concurrency Systems",
        executive_summary=sec_exec,
        sections=[sec_body],
        conclusion=sec_concl,
        limitations=["Small scale benchmark only"],
    )


@pytest.mark.asyncio
async def test_synthesis_engine_successful_synthesis(
    test_config: ResearchTeamConfig,
) -> None:
    budget = SharedRunBudget(config=test_config, max_total_tokens=10000, max_total_cost_usd=1.0)
    mock_llm = MagicMock()
    mock_report = _create_mock_report()

    mock_response = MagicMock()
    mock_response.error = None
    mock_response.structured_response = mock_report
    mock_response.usage = {"input_tokens": 500, "output_tokens": 300}
    mock_llm.complete_structure = AsyncMock(return_value=mock_response)

    engine = SynthesisEngine(
        config=test_config,
        llm_service=mock_llm,
        prompt_manager=PromptTemplateManager(config=test_config),
    )

    pool = AuthoritativeSourcePool()
    pool.add_source(
        SourceReference(
            source_id="src_1",
            chunk_id="chk_1",
            title="Concurrency Study",
            content="Multi-agent systems improve execution throughput.",
        )
    )

    worker_results = {
        "b1": WorkerExecutionResult(
            brief_id="b1",
            status=WorkerStatus.COMPLETED,
            output=WorkerResearchOutput(
                brief_id="b1",
                findings_summary="Concurrency findings summary",
                sources=[],
                claims=[],
            ),
        )
    }

    report = await engine.synthesize(
        topic="Multi-Agent Concurrency",
        worker_results=worker_results,
        source_pool=pool,
        shared_budget=budget,
    )

    assert report.topic == "Multi-Agent Concurrency"
    assert report.research_team_version == test_config.system_version
    assert any(b.brief_id == "b1" for b in report.brief_coverages)
    assert budget.current_tokens == 800
    assert budget.current_cost_usd > 0.0


@pytest.mark.asyncio
async def test_synthesis_engine_documents_failed_and_partial_workers(
    test_config: ResearchTeamConfig,
) -> None:
    mock_llm = MagicMock()
    mock_report = _create_mock_report()

    mock_response = MagicMock()
    mock_response.error = None
    mock_response.structured_response = mock_report
    mock_response.usage = {"input_tokens": 200, "output_tokens": 100}
    mock_llm.complete_structure = AsyncMock(return_value=mock_response)

    engine = SynthesisEngine(
        config=test_config,
        llm_service=mock_llm,
        prompt_manager=PromptTemplateManager(config=test_config),
    )

    pool = AuthoritativeSourcePool()
    pool.add_source(
        SourceReference(
            source_id="src_1",
            chunk_id="chk_1",
            title="Partial Doc",
            content="Some salvaged evidence.",
        )
    )

    worker_results = {
        "b_ok": WorkerExecutionResult(
            brief_id="b_ok",
            status=WorkerStatus.COMPLETED,
            output=WorkerResearchOutput(
                brief_id="b_ok",
                findings_summary="Good findings",
                sources=[],
                claims=[],
            ),
        ),
        "b_partial": WorkerExecutionResult(
            brief_id="b_partial",
            status=WorkerStatus.PARTIAL,
            error_message="Hit 5 step ceiling",
            output=WorkerResearchOutput(
                brief_id="b_partial",
                findings_summary="Partial findings",
                sources=[],
                claims=[],
            ),
        ),
        "b_fail": WorkerExecutionResult(
            brief_id="b_fail",
            status=WorkerStatus.FAILED,
            error_message="Connection timeout to API",
            output=None,
        ),
    }

    report = await engine.synthesize(
        topic="Multi-Agent Concurrency",
        worker_results=worker_results,
        source_pool=pool,
    )

    # Must document failed and partial workers in limitations
    assert any("b_partial" in lim for lim in report.unresolved_limitations)
    assert any("b_fail" in lim for lim in report.unresolved_limitations)
    assert any("b_partial" in lim for lim in report.limitations)
    assert any("b_fail" in lim for lim in report.limitations)


@pytest.mark.asyncio
async def test_synthesis_engine_empty_topic_raises(
    test_config: ResearchTeamConfig,
) -> None:
    engine = SynthesisEngine(config=test_config)
    with pytest.raises(SynthesisError, match="topic cannot be empty"):
        await engine.synthesize(
            topic="   ",
            worker_results={},
            source_pool=AuthoritativeSourcePool(),
        )


@pytest.mark.asyncio
async def test_synthesis_engine_all_workers_failed_and_zero_sources_raises(
    test_config: ResearchTeamConfig,
) -> None:
    engine = SynthesisEngine(config=test_config)
    worker_results = {
        "b_f": WorkerExecutionResult(
            brief_id="b_f",
            status=WorkerStatus.FAILED,
            error_message="Complete crash",
        )
    }
    with pytest.raises(SynthesisError, match="all parallel workers failed"):
        await engine.synthesize(
            topic="Topic",
            worker_results=worker_results,
            source_pool=AuthoritativeSourcePool(),
        )


@pytest.mark.asyncio
async def test_synthesis_engine_citation_reconciler(
    test_config: ResearchTeamConfig,
) -> None:
    mock_llm = MagicMock()
    # Report with an invalid citation and an inaccurate quotation
    bad_report = Report(
        topic="Testing Reconciler",
        title="Reconciler Test",
        executive_summary=ReportSection(
            section_id="sec_exec",
            title="Summary",
            content="Some prose.",
            claims=[
                Claim(
                    claim_id="c_valid_quote",
                    section_id="sec_exec",
                    text="Exact valid quote.",
                    citations=[
                        Citation(
                            citation_id="cite_1",
                            source_id="src_1",
                            chunk_id="chk_1",
                            citation_type=CitationType.QUOTATION,
                            quoted_passage="concurrency scaling",
                        ),
                        Citation(
                            citation_id="cite_fake",
                            source_id="fake_src",
                            chunk_id="fake_chk",
                            citation_type=CitationType.PARAPHRASE,
                        ),
                    ],
                ),
                Claim(
                    claim_id="c_invalid_quote",
                    section_id="sec_exec",
                    text="Inaccurate quote passage.",
                    citations=[
                        Citation(
                            citation_id="cite_2",
                            source_id="src_1",
                            chunk_id="chk_1",
                            citation_type=CitationType.QUOTATION,
                            quoted_passage="this string does not exist in source",
                        )
                    ],
                ),
            ],
        ),
        sections=[],
        conclusion=ReportSection(
            section_id="sec_concl",
            title="Conclusion",
            content="Done.",
            claims=[
                Claim(
                    claim_id="c_orphan",
                    section_id="sec_concl",
                    text="Claim with only fake citation.",
                    is_factual=True,
                    citations=[
                        Citation(
                            citation_id="cite_orphan",
                            source_id="nonexistent",
                            chunk_id="nonexistent",
                            citation_type=CitationType.PARAPHRASE,
                        )
                    ],
                )
            ],
        ),
    )

    mock_response = MagicMock()
    mock_response.error = None
    mock_response.structured_response = bad_report
    mock_response.usage = {"input_tokens": 100, "output_tokens": 50}
    mock_llm.complete_structure = AsyncMock(return_value=mock_response)

    engine = SynthesisEngine(
        config=test_config,
        llm_service=mock_llm,
        prompt_manager=PromptTemplateManager(config=test_config),
    )

    pool = AuthoritativeSourcePool(
        [
            SourceReference(
                source_id="src_1",
                chunk_id="chk_1",
                title="Real Source",
                content="This text discusses concurrency scaling in distributed workers.",
            )
        ]
    )

    report = await engine.synthesize(
        topic="Testing Reconciler",
        worker_results={
            "b1": WorkerExecutionResult(
                brief_id="b1",
                status=WorkerStatus.COMPLETED,
                output=WorkerResearchOutput(
                    brief_id="b1",
                    findings_summary="Findings",
                    sources=[],
                    claims=[],
                ),
            )
        },
        source_pool=pool,
    )

    c1 = report.executive_summary.claims[0]
    assert c1.section_id == "sec_exec"
    # cite_fake was pruned, only cite_1 remained
    assert len(c1.citations) == 1
    assert c1.citations[0].citation_id == "cite_1"
    assert c1.citations[0].citation_type == CitationType.QUOTATION

    c2 = report.executive_summary.claims[1]
    # Inaccurate quotation was downgraded to paraphrase
    assert c2.citations[0].citation_type == CitationType.PARAPHRASE
    assert c2.citations[0].quoted_passage is None

    # Orphan claim had its citation pruned and was downgraded from factual
    c_orphan = report.conclusion.claims[0]
    assert len(c_orphan.citations) == 0
    assert c_orphan.is_factual is False
