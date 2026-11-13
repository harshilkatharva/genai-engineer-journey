"""Unit tests for each individual Jinja2 prompt template and rendering methods."""

from __future__ import annotations

from self_improving_report_writer.models.critique import (
    Critique,
    CritiqueIssue,
    IssueCategory,
    IssueSeverity,
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
    DeterministicVerificationReport,
)
from self_improving_report_writer.prompts.manager import get_prompt_manager


def _build_test_report() -> Report:
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
        content="Forward outlook.",
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


def test_generation_templates() -> None:
    """Test generation system prompt and user prompt rendering."""
    mgr = get_prompt_manager()

    sys_prompt = mgr.get_system_prompt("generation", "system")
    assert "You are a rigorous, evidence-grounded research report writer." in sys_prompt
    assert "UNTRUSTED DATA BARRIER" in sys_prompt
    assert "CITATION DISCIPLINE" in sys_prompt

    sources = [
        SourceReference(
            source_id="src_1",
            chunk_id="chk_1",
            title="Quantum Report",
            content="Evidence text here.",
            score=0.9,
        )
    ]
    user_prompt = mgr.render_report_generation_prompt(
        topic="Quantum Computing",
        sources=sources,
        guidelines=["Focus on NISQ devices", "Include error mitigation"],
    )
    assert "RESEARCH TOPIC:\nQuantum Computing" in user_prompt
    assert "ADDITIONAL WRITING GUIDELINES TO OBSERVE:" in user_prompt
    assert "- Focus on NISQ devices" in user_prompt
    assert (
        '<untrusted_source_context id="chk_1" source_id="src_1" title="Quantum Report">'
        in user_prompt
    )
    assert "Evidence text here." in user_prompt


def test_revision_templates() -> None:
    """Test revision system prompt and user prompt rendering."""
    mgr = get_prompt_manager()

    sys_prompt = mgr.get_system_prompt("revision", "system")
    assert "You are an elite research report revision specialist." in sys_prompt
    assert "Critical Revision Rules:" in sys_prompt

    report = _build_test_report()
    sources = [
        SourceReference(
            source_id="src_1",
            chunk_id="chk_1",
            title="Benchmark",
            content="quantum speedup observed",
            score=0.9,
        )
    ]
    issue = CritiqueIssue(
        issue_id="ISSUE-01",
        category=IssueCategory.FACTUAL_GROUNDING,
        severity=IssueSeverity.MAJOR,
        affected_section_id="sec_exec",
        affected_claim_id="claim_1",
        explanation="Missing citation chunk.",
        suggested_correction="Add chunk reference.",
    )
    critique = Critique(round_number=1, issues=[issue], should_revise=True)

    user_prompt = mgr.render_revision_prompt(
        current_report=report,
        sources=sources,
        critique=critique,
    )
    assert "Please revise the research report below" in user_prompt
    assert "Quantum Optimization" in user_prompt
    assert "[MAJOR] (factual_grounding)" in user_prompt
    assert "Missing citation chunk." in user_prompt
    assert "Increment version to 2" in user_prompt


def test_judging_templates() -> None:
    """Test judging system prompt and user prompt rendering."""
    mgr = get_prompt_manager()

    sys_prompt = mgr.get_system_prompt("judging", "system")
    assert "You are an expert, impartial academic and technical evaluation judge." in sys_prompt
    assert "Calibrated Scoring:" in sys_prompt

    report = _build_test_report()
    rubric = get_default_rubric_config()
    sources = {
        "chk_1": SourceReference(
            source_id="src_1",
            chunk_id="chk_1",
            title="Benchmark",
            content="quantum speedup observed",
            score=0.9,
        )
    }
    det_report = DeterministicVerificationReport(
        passed=True,
        schema_valid=True,
        citations_valid=True,
        passages_matched=True,
        total_claims=1,
        total_citations=1,
    )

    user_prompt = mgr.render_rubric_judge_prompt(
        report=report,
        rubric_config=rubric,
        sources=sources,
        deterministic_report=det_report,
    )
    assert (
        "Please evaluate the following research report according to the specified rubric criteria."
        in user_prompt
    )
    assert "factual_accuracy" in user_prompt
    assert "Gate 1 Deterministic Verification Status: PASSED" in user_prompt
    assert "Quantum Optimization" in user_prompt


def test_verification_claim_entailment_templates() -> None:
    """Test claim entailment system prompt and user prompt rendering."""
    mgr = get_prompt_manager()

    sys_prompt = mgr.get_system_prompt("verification", "claim_entailment_system")
    assert (
        "You are an objective natural language inference (NLI) and fact-checking evaluator."
        in sys_prompt
    )
    assert "ENTAILS: The cited passage directly substantiates and supports the claim." in sys_prompt

    user_prompt = mgr.render_claim_entailment_prompt(
        claim_id="claim_001",
        claim_text="Quantum annealer reached 5,000 qubits.",
        cited_chunks=[("chk_1", "The analog processor operates with 5,000+ qubits.")],
    )
    assert "claim_001" in user_prompt
    assert "Quantum annealer reached 5,000 qubits." in user_prompt
    assert '<untrusted_source_context chunk_id="chk_1">' in user_prompt
    assert "5,000+ qubits" in user_prompt


def test_verification_batch_claim_entailment_template() -> None:
    """Test batch claim entailment user prompt rendering."""
    mgr = get_prompt_manager()

    user_prompt = mgr.render_batch_claim_entailment_prompt(
        section_id="sec_exec",
        section_title="Executive Summary",
        claims=[
            ("c1", "Claim one text.", ["chk_1"]),
            ("c2", "Claim two text.", ["chk_2"]),
        ],
        cited_chunks=[
            ("chk_1", "Content 1"),
            ("chk_2", "Content 2"),
        ],
    )
    assert "Executive Summary" in user_prompt
    assert "sec_exec" in user_prompt
    assert "- Claim ID: c1" in user_prompt
    assert "- Claim ID: c2" in user_prompt
    assert '<untrusted_source_context chunk_id="chk_1">' in user_prompt
    assert '<untrusted_source_context chunk_id="chk_2">' in user_prompt


def test_verification_prose_consistency_templates() -> None:
    """Test prose consistency system prompt and user prompt rendering."""
    mgr = get_prompt_manager()

    sys_prompt = mgr.get_system_prompt("verification", "prose_consistency_system")
    assert "You are a technical document consistency auditor." in sys_prompt
    assert "unstructured_claims" in sys_prompt

    user_prompt = mgr.render_prose_consistency_prompt(
        section_id="sec_exec",
        section_title="Executive Summary",
        prose_content="Quantum optimization is rapidly advancing.",
        structured_claims=["Quantum annealing demonstrated 3.2x speedup."],
    )
    assert "sec_exec" in user_prompt
    assert "Executive Summary" in user_prompt
    assert "Quantum optimization is rapidly advancing." in user_prompt
    assert "- Quantum annealing demonstrated 3.2x speedup." in user_prompt
