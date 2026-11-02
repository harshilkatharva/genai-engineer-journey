"""Unit tests for SemanticGroundingVerifier: entailment, prose consistency, and defect tagging."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from genai_core import LLMError, LLMManagerResponse

from self_improving_report_writer.core.exceptions import VerificationError
from self_improving_report_writer.judging.prompts import (
    ClaimEntailmentEvaluation,
    ProseConsistencyEvaluation,
)
from self_improving_report_writer.models.report import (
    Citation,
    CitationType,
    Claim,
    Report,
    ReportSection,
    SourceReference,
)
from self_improving_report_writer.models.verification import DefectSeverity
from self_improving_report_writer.verification.semantic_grounding_verifier import (
    SemanticGroundingVerifier,
)


@pytest.fixture
def test_report() -> Report:
    """Create a minimal report for grounding evaluation."""
    return Report(
        topic="Quantum Computing",
        title="Quantum Annealing Overview",
        executive_summary=ReportSection(
            section_id="sec_exec",
            title="Executive Summary",
            content="Quantum annealing demonstrated speedup on logistics graphs. [cite_1]",
            claims=[
                Claim(
                    claim_id="claim_1",
                    section_id="sec_exec",
                    text="Quantum annealing demonstrated speedup on logistics graphs.",
                    citations=[
                        Citation(
                            citation_id="cite_1",
                            source_id="src_quantum_01",
                            chunk_id="chunk_q01",
                            citation_type=CitationType.PARAPHRASE,
                        )
                    ],
                )
            ],
        ),
        sections=[
            ReportSection(
                section_id="sec_analysis",
                title="Analysis",
                content="Analog quantum annealers operate with 5,000+ qubits. [cite_2]",
                claims=[
                    Claim(
                        claim_id="claim_2",
                        section_id="sec_analysis",
                        text="Analog quantum annealers operate with 5,000+ qubits.",
                        citations=[
                            Citation(
                                citation_id="cite_2",
                                source_id="src_quantum_02",
                                chunk_id="chunk_q02",
                                citation_type=CitationType.PARAPHRASE,
                            )
                        ],
                    )
                ],
            )
        ],
        conclusion=ReportSection(
            section_id="sec_conc",
            title="Conclusion",
            content="Quantum technology is advancing rapidly.",
            claims=[],
        ),
    )


@pytest.mark.asyncio
async def test_grounding_verifier_entails(
    test_report: Report, sample_sources: list[SourceReference]
) -> None:
    """Verify that ENTAILS verdicts yield 1.0 scores and pass Gate 2."""
    mock_llm = AsyncMock()

    entails_eval = ClaimEntailmentEvaluation(
        entailment="ENTAILS",
        score=1.0,
        reasoning="Directly supported by the source chunk text.",
    )
    prose_eval = ProseConsistencyEvaluation(
        unstructured_claims=[],
        reasoning="All factual assertions are covered in structured claims.",
    )

    # Return entailment for 2 claims, then prose consistency checks for 3 sections
    mock_llm.complete_structure.side_effect = [
        LLMManagerResponse(
            structured_response=entails_eval,
            usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            raw_response={"cost": 0.0005},
        ),
        LLMManagerResponse(
            structured_response=prose_eval,
            usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            raw_response={"cost": 0.0005},
        ),
        LLMManagerResponse(
            structured_response=entails_eval,
            usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            raw_response={"cost": 0.0005},
        ),
        LLMManagerResponse(
            structured_response=prose_eval,
            usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            raw_response={"cost": 0.0005},
        ),
        LLMManagerResponse(
            structured_response=prose_eval,
            usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            raw_response={"cost": 0.0005},
        ),
    ]

    verifier = SemanticGroundingVerifier(llm_service=mock_llm)
    res = await verifier.verify_grounding(test_report, sample_sources)

    assert res.passed is True
    assert res.all_claims_grounded is True
    assert res.grounding_score == 1.0
    assert res.critical_count == 0
    assert res.major_count == 0
    assert len(res.issues) == 0
    assert len(res.claim_evaluations) == 2
    assert res.tokens_used > 0
    assert res.cost_usd > 0.0


@pytest.mark.asyncio
async def test_grounding_verifier_neutral_flags_major(
    test_report: Report, sample_sources: list[SourceReference]
) -> None:
    """Verify that NEUTRAL entailment triggers a MAJOR defect."""
    mock_llm = AsyncMock()

    neutral_eval = ClaimEntailmentEvaluation(
        entailment="NEUTRAL",
        score=0.5,
        reasoning="The source mentions batteries but does not discuss high energy density.",
    )

    mock_llm.complete_structure.return_value = LLMManagerResponse(
        structured_response=neutral_eval,
    )

    verifier = SemanticGroundingVerifier(llm_service=mock_llm, check_prose_consistency=False)
    res = await verifier.verify_grounding(test_report, sample_sources)

    assert res.passed is False
    assert res.all_claims_grounded is False
    assert res.major_count == 2
    assert all(i.severity == DefectSeverity.MAJOR for i in res.issues)


@pytest.mark.asyncio
async def test_grounding_verifier_contradicts_flags_critical(
    test_report: Report, sample_sources: list[SourceReference]
) -> None:
    """Verify that CONTRADICTS entailment triggers a CRITICAL defect."""
    mock_llm = AsyncMock()

    contradict_eval = ClaimEntailmentEvaluation(
        entailment="CONTRADICTS",
        score=0.0,
        reasoning="The source explicitly contradicts the claim.",
    )

    mock_llm.complete_structure.return_value = LLMManagerResponse(
        structured_response=contradict_eval,
    )

    verifier = SemanticGroundingVerifier(llm_service=mock_llm, check_prose_consistency=False)
    res = await verifier.verify_grounding(test_report, sample_sources)

    assert res.passed is False
    assert res.critical_count == 2
    assert all(i.severity == DefectSeverity.CRITICAL for i in res.issues)
    assert res.grounding_score == 0.0


@pytest.mark.asyncio
async def test_grounding_verifier_uncited_claim_flags_major(
    sample_sources: list[SourceReference],
) -> None:
    """Verify that factual claims without citations trigger MAJOR defect without LLM call."""
    sources_dict = {s.source_id: s for s in sample_sources}
    mock_llm = AsyncMock()

    report = Report(
        topic="AI Safety",
        title="AI Alignment",
        executive_summary=ReportSection(
            section_id="sec_exec",
            title="Summary",
            content="RLHF reduces harmful outputs.",
            claims=[
                Claim(
                    claim_id="claim_uncited",
                    section_id="sec_exec",
                    text="RLHF reduces harmful outputs by 90%.",
                    citations=[],  # Empty citations on factual claim
                )
            ],
        ),
        sections=[
            ReportSection(
                section_id="sec_1",
                title="Body",
                content="General discussion of alignment techniques.",
                claims=[],
            )
        ],
        conclusion=ReportSection(
            section_id="sec_conc",
            title="Conclusion",
            content="Alignment remains an active research area.",
            claims=[],
        ),
    )

    verifier = SemanticGroundingVerifier(llm_service=mock_llm, check_prose_consistency=False)
    res = await verifier.verify_grounding(report, sources_dict)

    assert res.passed is False
    assert res.major_count == 1
    assert res.issues[0].severity == DefectSeverity.MAJOR
    assert "lacks supporting citations" in res.issues[0].message
    assert mock_llm.complete_structure.call_count == 0


@pytest.mark.asyncio
async def test_grounding_verifier_unknown_chunk_flags_major(
    sample_sources: list[SourceReference],
) -> None:
    """Verify that citations to non-existent chunks trigger MAJOR defect."""
    sources_dict = {s.source_id: s for s in sample_sources}
    mock_llm = AsyncMock()

    report = Report(
        topic="AI Safety",
        title="AI Alignment",
        executive_summary=ReportSection(
            section_id="sec_exec",
            title="Summary",
            content="RLHF reduces harmful outputs. [cite_unknown]",
            claims=[
                Claim(
                    claim_id="claim_unknown",
                    section_id="sec_exec",
                    text="RLHF reduces harmful outputs.",
                    citations=[
                        Citation(
                            citation_id="cite_unknown",
                            source_id="nonexistent_src",
                            chunk_id="nonexistent_chunk",
                            citation_type=CitationType.PARAPHRASE,
                        )
                    ],
                )
            ],
        ),
        sections=[
            ReportSection(
                section_id="sec_1",
                title="Body",
                content="General discussion.",
                claims=[],
            )
        ],
        conclusion=ReportSection(
            section_id="sec_conc",
            title="Conclusion",
            content="Alignment remains active.",
            claims=[],
        ),
    )

    verifier = SemanticGroundingVerifier(llm_service=mock_llm, check_prose_consistency=False)
    res = await verifier.verify_grounding(report, sources_dict)

    assert res.passed is False
    assert res.major_count == 1
    assert "unknown chunks" in res.issues[0].message
    assert mock_llm.complete_structure.call_count == 0


@pytest.mark.asyncio
async def test_grounding_verifier_prose_consistency_defect(
    test_report: Report, sample_sources: list[SourceReference]
) -> None:
    """Verify that ungrounded factual assertions in prose trigger MAJOR defect."""
    mock_llm = AsyncMock()

    entails_eval = ClaimEntailmentEvaluation(
        entailment="ENTAILS",
        score=1.0,
        reasoning="Supported.",
    )
    # Prose consistency flags an uncaptured factual assertion
    prose_eval_with_defect = ProseConsistencyEvaluation(
        unstructured_claims=["Solid state battery factory cost was $2.5 billion in 2024."],
        reasoning="The cost metric is stated as fact in prose but has no corresponding claim.",
    )
    clean_prose_eval = ProseConsistencyEvaluation(
        unstructured_claims=[],
        reasoning="All factual assertions covered.",
    )

    mock_llm.complete_structure.side_effect = [
        LLMManagerResponse(structured_response=entails_eval),
        LLMManagerResponse(structured_response=prose_eval_with_defect),
        LLMManagerResponse(structured_response=entails_eval),
        LLMManagerResponse(structured_response=clean_prose_eval),
        LLMManagerResponse(structured_response=clean_prose_eval),
    ]

    verifier = SemanticGroundingVerifier(llm_service=mock_llm, check_prose_consistency=True)
    res = await verifier.verify_grounding(test_report, sample_sources)

    assert res.passed is False
    assert res.major_count == 1
    assert any("ungrounded factual assertion in prose" in i.message for i in res.issues)


@pytest.mark.asyncio
async def test_grounding_verifier_llm_error_raises_exception(
    test_report: Report, sample_sources: list[SourceReference]
) -> None:
    """Verify that unexpected LLM failure raises VerificationError."""
    mock_llm = AsyncMock()

    mock_llm.complete_structure.return_value = LLMManagerResponse(
        error=LLMError(provider="google", code="unavailable", message="Service unavailable 503")
    )

    verifier = SemanticGroundingVerifier(llm_service=mock_llm)

    with pytest.raises(VerificationError, match="Claim entailment check failed"):
        await verifier.verify_grounding(test_report, sample_sources)
