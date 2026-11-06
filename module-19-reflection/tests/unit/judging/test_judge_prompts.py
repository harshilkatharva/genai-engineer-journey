"""Unit tests for judge and verifier prompt builders in judging/prompts.py."""

from __future__ import annotations

from self_improving_report_writer.judging.prompts import (
    build_batch_claim_entailment_prompt,
    build_claim_entailment_prompt,
    build_prose_consistency_prompt,
    build_rubric_judge_prompt,
    escape_xml_context,
)
from self_improving_report_writer.models.report import (
    Citation,
    CitationType,
    Claim,
    Report,
    ReportSection,
    SourceReference,
)
from self_improving_report_writer.models.rubric import get_default_rubric_config
from self_improving_report_writer.models.verification import (
    DefectCategory,
    DefectSeverity,
    DeterministicVerificationReport,
    VerificationIssue,
)


def _build_simple_report() -> Report:
    cite = Citation(
        citation_id="cite_1",
        source_id="src_1",
        chunk_id="chk_1",
        citation_type=CitationType.QUOTATION,
        quoted_passage="quantum speedup observed",
    )
    claim = Claim(
        claim_id="claim_1",
        section_id="sec_exec",
        text="Quantum annealing demonstrated 3.2x speedup.",
        is_factual=True,
        citations=[cite],
    )
    exec_sec = ReportSection(
        section_id="sec_exec",
        title="Executive Summary",
        content="Overview with citation [cite_1].",
        claims=[claim],
    )
    concl_sec = ReportSection(
        section_id="sec_concl",
        title="Conclusion",
        content="Conclusion content.",
        claims=[],
    )
    return Report(
        topic="Quantum Optimization",
        title="Report on Quantum Optimization",
        executive_summary=exec_sec,
        sections=[],
        conclusion=concl_sec,
        version=1,
    )


def test_escape_xml_context() -> None:
    """Test XML character escaping prevents delimiter breakout."""
    raw = 'Text with <tag> & "quotes" > 100'
    escaped = escape_xml_context(raw)
    assert "&amp;" in escaped
    assert "&lt;tag&gt;" in escaped
    assert "<" not in escaped
    assert ">" not in escaped
    assert raw != escaped

    plain = "Clean simple text."
    assert escape_xml_context(plain) == plain


def test_build_rubric_judge_prompt_with_deterministic_report() -> None:
    """Test build_rubric_judge_prompt properly includes criteria, report, sources, and verification report."""
    report = _build_simple_report()
    rubric = get_default_rubric_config()

    sources = {
        "chk_1": SourceReference(
            source_id="src_1",
            chunk_id="chk_1",
            title="Quantum Benchmark 2026",
            content="In tests, quantum speedup observed was 3.2x <injection> & note.",
            score=0.95,
        )
    }

    det_rep = DeterministicVerificationReport(
        passed=False,
        schema_valid=True,
        citations_valid=True,
        passages_matched=False,
        total_claims=1,
        total_citations=1,
        critical_count=1,
        major_count=0,
        minor_count=0,
        issues=[
            VerificationIssue(
                issue_id="ISSUE-01",
                category=DefectCategory.CITATION_INTEGRITY,
                severity=DefectSeverity.CRITICAL,
                message="Quotation substring mismatch.",
            )
        ],
    )

    prompt = build_rubric_judge_prompt(
        report=report,
        rubric_config=rubric,
        sources=sources,
        deterministic_report=det_rep,
    )

    # Validate criteria included
    assert "factual_accuracy" in prompt
    assert "evidence_grounding" in prompt
    assert "completeness" in prompt

    # Validate report topic and content
    assert "Quantum Optimization" in prompt
    assert "Report on Quantum Optimization" in prompt

    # Validate source context XML isolation and escaping
    assert (
        '<untrusted_source_context id="chk_1" source_id="src_1" title="Quantum Benchmark 2026">'
        in prompt
    )
    assert "&lt;injection&gt; &amp; note." in prompt
    assert "</untrusted_source_context>" in prompt

    # Validate deterministic report context
    assert "Gate 1 Deterministic Verification Status: FAILED" in prompt
    assert "Quotation substring mismatch." in prompt


def test_build_rubric_judge_prompt_without_sources_and_det_report() -> None:
    """Test build_rubric_judge_prompt when sources and deterministic report are omitted."""
    report = _build_simple_report()
    rubric = get_default_rubric_config()

    prompt = build_rubric_judge_prompt(
        report=report,
        rubric_config=rubric,
        sources={},
        deterministic_report=None,
    )

    assert "No sources provided." in prompt
    assert "Gate 1 Deterministic Verification Status: Not provided." in prompt


def test_build_claim_entailment_prompt() -> None:
    """Test build_claim_entailment_prompt creates well-formed prompt with sanitized chunks."""
    prompt = build_claim_entailment_prompt(
        claim_id="claim_001",
        claim_text="Quantum annealer uses 5,000+ qubits.",
        cited_chunks=[("chk_1", "The analog processor operates with <5000+> qubits & low noise.")],
    )

    assert "claim_001" in prompt
    assert "Quantum annealer uses 5,000+ qubits." in prompt
    assert '<untrusted_source_context chunk_id="chk_1">' in prompt
    assert "&lt;5000+&gt; qubits &amp; low noise." in prompt
    assert "</untrusted_source_context>" in prompt


def test_build_claim_entailment_prompt_empty_chunks() -> None:
    """Test build_claim_entailment_prompt handles empty chunks."""
    prompt = build_claim_entailment_prompt(
        claim_id="claim_002",
        claim_text="Ungrounded claim text.",
        cited_chunks=[],
    )
    assert "NO CITED CHUNKS PROVIDED." in prompt


def test_build_batch_claim_entailment_prompt() -> None:
    """Test build_batch_claim_entailment_prompt with multiple claims and deduplicated source chunks."""
    claims = [
        ("c1", "Claim one statement.", ["chk_1"]),
        ("c2", "Claim two statement.", ["chk_1", "chk_2"]),
    ]
    chunks = [
        ("chk_1", "Source text one with <tag>."),
        ("chk_2", "Source text two & extra."),
    ]

    prompt = build_batch_claim_entailment_prompt(
        section_id="sec_1",
        section_title="Findings",
        claims=claims,
        cited_chunks=chunks,
    )

    assert "Findings" in prompt
    assert "sec_1" in prompt
    assert "- Claim ID: c1" in prompt
    assert "- Claim ID: c2" in prompt
    assert '<untrusted_source_context chunk_id="chk_1">' in prompt
    assert "&lt;tag&gt;" in prompt
    assert '<untrusted_source_context chunk_id="chk_2">' in prompt
    assert "&amp; extra." in prompt


def test_build_batch_claim_entailment_prompt_empty() -> None:
    """Test build_batch_claim_entailment_prompt with empty claims and chunks."""
    prompt = build_batch_claim_entailment_prompt(
        section_id="sec_empty",
        section_title="Empty Section",
        claims=[],
        cited_chunks=[],
    )
    assert "NO CLAIMS TO EVALUATE." in prompt
    assert "NO CITED CHUNKS PROVIDED." in prompt


def test_build_prose_consistency_prompt() -> None:
    """Test build_prose_consistency_prompt formatting with structured claims."""
    prompt = build_prose_consistency_prompt(
        section_id="sec_exec",
        section_title="Executive Summary",
        prose_content="Quantum computing shows great promise across industries.",
        structured_claims=["Quantum annealing demonstrated 3.2x speedup."],
    )

    assert "sec_exec" in prompt
    assert "Executive Summary" in prompt
    assert "Quantum computing shows great promise across industries." in prompt
    assert "- Quantum annealing demonstrated 3.2x speedup." in prompt


def test_build_prose_consistency_prompt_empty_claims() -> None:
    """Test build_prose_consistency_prompt when structured claims list is empty."""
    prompt = build_prose_consistency_prompt(
        section_id="sec_exec",
        section_title="Executive Summary",
        prose_content="Prose without claims.",
        structured_claims=[],
    )
    assert "No structured claims defined for this section." in prompt
