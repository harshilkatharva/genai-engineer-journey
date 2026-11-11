"""Regression tests verifying source_id attribution in judge prompts (FINDING-08)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from genai_core import LLMManagerResponse

from self_improving_report_writer.judging.prompts import (
    LLMJudgeEvaluation,
    build_rubric_judge_prompt,
)
from self_improving_report_writer.judging.rubric_judge import RubricJudge
from self_improving_report_writer.models.report import (
    Citation,
    CitationType,
    Claim,
    Report,
    ReportSection,
    SourceReference,
)
from self_improving_report_writer.models.rubric import DimensionScore, get_default_rubric_config


def _make_report_with_citation(source_id: str, chunk_id: str) -> Report:
    claim = Claim(
        claim_id="claim_1",
        section_id="sec_exec",
        text="Solar capacity expanded by 35% year over year.",
        citations=[
            Citation(
                citation_id="cite_1",
                source_id=source_id,
                chunk_id=chunk_id,
                citation_type=CitationType.QUOTATION,
                quoted_passage="Solar capacity expanded by 35% year over year.",
            )
        ],
    )
    return Report(
        topic="Renewable Energy Outlook",
        title="Renewable Energy Outlook",
        executive_summary=ReportSection(
            section_id="sec_exec",
            title="Executive Summary",
            content="Solar capacity expanded by 35% year over year.",
            claims=[claim],
        ),
        sections=[],
        conclusion=ReportSection(
            section_id="sec_conc",
            title="Conclusion",
            content="Growth continues.",
            claims=[],
        ),
    )


def test_source_id_included_in_judge_prompt_tags() -> None:
    """Chunk-keyed source dicts must still expose source_id attribute in judge XML tags."""
    report = _make_report_with_citation(source_id="src_doc_42", chunk_id="chunk_0")
    sources = {
        "chunk_0": SourceReference(
            source_id="src_doc_42",
            chunk_id="chunk_0",
            title="IEA Annual Solar Review",
            content="Solar capacity expanded by 35% year over year across key markets.",
            score=0.96,
        )
    }

    prompt = build_rubric_judge_prompt(
        report=report,
        rubric_config=get_default_rubric_config(),
        sources=sources,
    )

    assert (
        '<untrusted_source_context id="chunk_0" source_id="src_doc_42" title="IEA Annual Solar Review">'
        in prompt
    )
    assert '"source_id": "src_doc_42"' in prompt


def test_multiple_chunks_preserve_distinct_source_ids_in_judge_prompt() -> None:
    """Each source chunk tag must carry its own distinct source_id for attribution verification."""
    report = _make_report_with_citation(source_id="src_alpha", chunk_id="chk_a")
    sources = {
        "chk_a": SourceReference(
            source_id="src_alpha",
            chunk_id="chk_a",
            title="Alpha Report",
            content="Solar capacity expanded by 35% year over year.",
        ),
        "chk_b": SourceReference(
            source_id="src_beta",
            chunk_id="chk_b",
            title="Beta Report",
            content="Wind power added 120 GW globally.",
        ),
    }

    prompt = build_rubric_judge_prompt(
        report=report,
        rubric_config=get_default_rubric_config(),
        sources=sources,
    )

    assert (
        '<untrusted_source_context id="chk_a" source_id="src_alpha" title="Alpha Report">' in prompt
    )
    assert (
        '<untrusted_source_context id="chk_b" source_id="src_beta" title="Beta Report">' in prompt
    )


@pytest.mark.asyncio
async def test_rubric_judge_evaluate_forwards_source_id_to_llm_prompt() -> None:
    """RubricJudge.evaluate must send XML source blocks containing source_id to the judge LLM."""
    report = _make_report_with_citation(source_id="src_doc_42", chunk_id="chunk_0")
    sources = {
        "chunk_0": SourceReference(
            source_id="src_doc_42",
            chunk_id="chunk_0",
            title="IEA Annual Solar Review",
            content="Solar capacity expanded by 35% year over year.",
        )
    }

    rubric = get_default_rubric_config()
    raw_eval = LLMJudgeEvaluation(
        dimension_scores=[
            DimensionScore(
                dimension_name=c.name,
                score=0.92,
                passed=True,
                reasoning="Supported by cited source.",
            )
            for c in rubric.criteria
        ],
        overall_feedback="Accurate and properly cited.",
    )
    mock_llm = AsyncMock()
    mock_llm.complete_structure.return_value = LLMManagerResponse(
        structured_response=raw_eval,
        usage={"input_tokens": 300, "output_tokens": 120, "total_tokens": 420},
        raw_response={"cost": 0.001},
    )

    judge = RubricJudge(
        llm_service=mock_llm,
        rubric_config=rubric,
        judge_model="gemini-3.1-flash-lite",
        generator_model="gpt-4o-mini",
    )

    await judge.evaluate(report, sources)

    sent_prompt = mock_llm.complete_structure.call_args.kwargs["prompt"]
    assert (
        '<untrusted_source_context id="chunk_0" source_id="src_doc_42" title="IEA Annual Solar Review">'
        in sent_prompt
    )
