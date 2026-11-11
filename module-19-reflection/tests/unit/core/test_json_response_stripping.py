"""Unit tests for Markdown fence stripping and JSON unwrapping across LLM callers (FINDING-10)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from genai_core import LLMManagerResponse

from self_improving_report_writer.core.exceptions import GenerationError, ReflectionError
from self_improving_report_writer.core.json_parsing import (
    parse_json_with_fence_stripping,
    strip_markdown_json_fences,
    unwrap_llm_response_payload,
)
from self_improving_report_writer.generation.report_generator import ReportGenerator
from self_improving_report_writer.judging.rubric_judge import RubricJudge
from self_improving_report_writer.models.critique import Critique
from self_improving_report_writer.models.llm_schemas import (
    ClaimEntailmentEvaluation,
    LLMJudgeEvaluation,
    ProseConsistencyEvaluation,
    SectionClaimsEntailmentEvaluation,
    SingleClaimEntailmentVerdict,
)
from self_improving_report_writer.models.report import Report, SourceReference
from self_improving_report_writer.models.rubric import DimensionScore, get_default_rubric_config
from self_improving_report_writer.reflection.revision_engine import RevisionEngine
from self_improving_report_writer.verification.semantic_grounding_verifier import (
    SemanticGroundingVerifier,
)


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ('{"key": "value"}', '{"key": "value"}'),
        ('```json\n{"key": "value"}\n```', '{"key": "value"}'),
        ('  ```JSON\n  {"key": "value"}\n  ```  ', '{"key": "value"}'),
        ('```\n{"key": "value"}\n```', '{"key": "value"}'),
        ('```json{"key": "value"}```', '{"key": "value"}'),
        ('Here is the JSON:\n```json\n{"key": "value"}\n```\nHope this helps.', '{"key": "value"}'),
        ('{"code": "```python\\nprint(1)\\n```"}', '{"code": "```python\\nprint(1)\\n```"}'),
        (
            '```json\n{"code": "```python\\nprint(1)\\n```"}\n```',
            '{"code": "```python\\nprint(1)\\n```"}',
        ),
        ("   ", ""),
    ],
)
def test_strip_markdown_json_fences_handles_all_fence_variants(
    raw_text: str,
    expected: str,
) -> None:
    """strip_markdown_json_fences should clean fences without corrupting raw JSON or nested backticks."""
    assert strip_markdown_json_fences(raw_text) == expected


def test_parse_and_unwrap_llm_response_payload_strips_fences() -> None:
    """parse_json_with_fence_stripping and unwrap_llm_response_payload should decode fenced JSON."""
    fenced = '```json\n{"status": "ok", "count": 2}\n```'
    assert parse_json_with_fence_stripping(fenced) == {"status": "ok", "count": 2}

    resp_text = LLMManagerResponse(structured_response=None, text=fenced)
    assert unwrap_llm_response_payload(resp_text) == {"status": "ok", "count": 2}

    resp_str_structured = LLMManagerResponse(
        structured_response='```\n{"nested": "```code```"}\n```',
        text=None,
    )
    assert unwrap_llm_response_payload(resp_str_structured) == {"nested": "```code```"}

    resp_empty = LLMManagerResponse(structured_response=None, text="   ")
    assert unwrap_llm_response_payload(resp_empty) is None


async def test_report_generator_strips_markdown_fences_from_text_response(
    valid_report: Report,
    sample_sources: list[SourceReference],
) -> None:
    """ReportGenerator should successfully parse Markdown-fenced JSON from response.text."""
    mock_llm = AsyncMock()
    fenced_json = f"```json\n{valid_report.model_dump_json()}\n```"
    mock_llm.complete_structure.return_value = LLMManagerResponse(
        structured_response=None,
        text=fenced_json,
        usage={"total_tokens": 1200},
    )

    generator = ReportGenerator(llm_service=mock_llm)
    report, tokens, _cost = await generator.generate_report(
        topic=valid_report.topic,
        sources=sample_sources,
    )

    assert isinstance(report, Report)
    assert report.title == valid_report.title
    assert tokens == 1200


async def test_report_generator_raises_on_invalid_fenced_json(
    sample_sources: list[SourceReference],
) -> None:
    """ReportGenerator should still raise GenerationError if fenced payload is invalid JSON."""
    mock_llm = AsyncMock()
    mock_llm.complete_structure.return_value = LLMManagerResponse(
        structured_response=None,
        text="```json\n{not valid json}\n```",
    )

    generator = ReportGenerator(llm_service=mock_llm)
    with pytest.raises(GenerationError, match="Report JSON validation failed"):
        await generator.generate_report(topic="Quantum", sources=sample_sources)


async def test_revision_engine_strips_markdown_fences_from_text_response(
    valid_report: Report,
    sample_sources: list[SourceReference],
) -> None:
    """RevisionEngine should parse Markdown-fenced JSON text responses before json loading."""
    revised_payload = valid_report.model_dump(mode="json")
    revised_payload["version"] = 2

    mock_llm = AsyncMock()
    mock_llm.complete_structure.return_value = LLMManagerResponse(
        structured_response=None,
        text=f"```\n{json.dumps(revised_payload)}\n```",
        usage={"total_tokens": 900},
    )

    engine = RevisionEngine(llm_service=mock_llm)
    critique = Critique(round_number=1, issues=[], summary="Clean", should_revise=False)
    report, tokens, _cost = await engine.revise_report(
        current_report=valid_report,
        sources=sample_sources,
        critique=critique,
    )

    assert report.version == 2
    assert report.created_by == "revision_llm"
    assert tokens == 900


async def test_revision_engine_raises_on_invalid_fenced_json(
    valid_report: Report,
    sample_sources: list[SourceReference],
) -> None:
    """RevisionEngine should raise ReflectionError on malformed fenced JSON."""
    mock_llm = AsyncMock()
    mock_llm.complete_structure.return_value = LLMManagerResponse(
        structured_response=None,
        text="```json\n{invalid\n```",
    )

    engine = RevisionEngine(llm_service=mock_llm)
    critique = Critique(round_number=1, issues=[], summary="Clean", should_revise=False)
    with pytest.raises(ReflectionError, match="Failed to parse text response as Report"):
        await engine.revise_report(
            current_report=valid_report,
            sources=sample_sources,
            critique=critique,
        )


async def test_rubric_judge_strips_markdown_fences_from_text_response(
    valid_report: Report,
    sample_sources: list[SourceReference],
) -> None:
    """RubricJudge should accept Markdown-fenced JSON responses when structured_response is None."""
    rubric = get_default_rubric_config()
    raw_eval = LLMJudgeEvaluation(
        dimension_scores=[
            DimensionScore(
                dimension_name=c.name,
                score=0.95,
                passed=True,
                reasoning="Strong adherence.",
            )
            for c in rubric.criteria
        ],
        overall_feedback="High quality report.",
    )
    mock_llm = AsyncMock()
    mock_llm.complete_structure.return_value = LLMManagerResponse(
        structured_response=None,
        text=f"```json\n{raw_eval.model_dump_json()}\n```",
        model="gemini-3.1-flash-lite",
    )

    judge = RubricJudge(
        llm_service=mock_llm,
        judge_model="gemini-3.1-flash-lite",
        generator_model="gemini-3.5-flash-lite",
    )
    sources_dict = {s.chunk_id: s for s in sample_sources}
    verdict = await judge.evaluate(valid_report, sources_dict)

    assert verdict.passed is True
    assert verdict.composite_score == 0.95


async def test_semantic_grounding_verifier_strips_markdown_fences_from_text_responses(
    valid_report: Report,
    sample_sources: list[SourceReference],
) -> None:
    """SemanticGroundingVerifier should parse Markdown-fenced JSON across batch and prose checks."""
    batch_exec = SectionClaimsEntailmentEvaluation(
        evaluations=[
            SingleClaimEntailmentVerdict(
                claim_id="claim_exec_1",
                entailment="ENTAILS",
                score=1.0,
                reasoning="Directly entailed.",
            )
        ]
    )
    batch_findings = SectionClaimsEntailmentEvaluation(
        evaluations=[
            SingleClaimEntailmentVerdict(
                claim_id="claim_find_1",
                entailment="ENTAILS",
                score=1.0,
                reasoning="Directly entailed.",
            ),
            SingleClaimEntailmentVerdict(
                claim_id="claim_find_2",
                entailment="ENTAILS",
                score=1.0,
                reasoning="Directly entailed.",
            ),
        ]
    )
    batch_concl = SectionClaimsEntailmentEvaluation(
        evaluations=[
            SingleClaimEntailmentVerdict(
                claim_id="claim_concl_1",
                entailment="ENTAILS",
                score=1.0,
                reasoning="Directly entailed.",
            )
        ]
    )
    prose_ok = ProseConsistencyEvaluation(
        unstructured_claims=[],
        reasoning="All factual assertions match structured claims.",
    )

    # Order of calls across sections: batch_exec, prose_exec, batch_findings, prose_findings, batch_concl, prose_concl
    side_effects = [
        LLMManagerResponse(
            structured_response=None, text=f"```json\n{batch_exec.model_dump_json()}\n```"
        ),
        LLMManagerResponse(
            structured_response=None, text=f"```\n{prose_ok.model_dump_json()}\n```"
        ),
        LLMManagerResponse(
            structured_response=None, text=f"```JSON\n{batch_findings.model_dump_json()}\n```"
        ),
        LLMManagerResponse(
            structured_response=None, text=f"```json\n{prose_ok.model_dump_json()}\n```"
        ),
        LLMManagerResponse(
            structured_response=None, text=f"```json\n{batch_concl.model_dump_json()}\n```"
        ),
        LLMManagerResponse(
            structured_response=None, text=f"```json\n{prose_ok.model_dump_json()}\n```"
        ),
    ]

    mock_llm = AsyncMock()
    mock_llm.complete_structure.side_effect = side_effects

    verifier = SemanticGroundingVerifier(
        llm_service=mock_llm,
        batch_claim_evaluation=True,
        check_prose_consistency=True,
    )
    result = await verifier.verify_grounding(valid_report, sample_sources)

    assert result.passed is True
    assert result.all_claims_grounded is True
    assert result.grounding_score == 1.0

    # Also verify sequential (non-batch) entailment mode with fenced text response
    single_ok = ClaimEntailmentEvaluation(
        entailment="ENTAILS",
        score=1.0,
        reasoning="Entailed.",
    )
    mock_llm_seq = AsyncMock()
    mock_llm_seq.complete_structure.return_value = LLMManagerResponse(
        structured_response=None,
        text=f"```json\n{single_ok.model_dump_json()}\n```",
    )
    verifier_seq = SemanticGroundingVerifier(
        llm_service=mock_llm_seq,
        batch_claim_evaluation=False,
        check_prose_consistency=False,
    )
    result_seq = await verifier_seq.verify_grounding(valid_report, sample_sources)
    assert result_seq.passed is True
    assert result_seq.grounding_score == 1.0
