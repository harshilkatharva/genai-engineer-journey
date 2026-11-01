"""Unit tests for DeterministicVerifier and text normalization."""

from __future__ import annotations

from self_improving_report_writer.models.report import (
    Citation,
    CitationType,
    Claim,
    Report,
    ReportSection,
    SourceReference,
)
from self_improving_report_writer.models.verification import (
    DefectCategory,
    DefectSeverity,
)
from self_improving_report_writer.verification.deterministic_verifier import (
    DeterministicVerifier,
    normalize_for_matching,
)


def test_normalize_for_matching() -> None:
    """Normalization should lowercase, collapse whitespace, and standardize quotes."""
    raw = "  “Quantum\n\tAnnealing”   isn’t  ‘classical’\xa0computing.  "
    expected = "\"quantum annealing\" isn't 'classical' computing."
    assert normalize_for_matching(raw) == expected
    assert normalize_for_matching("") == ""


def test_verifier_passes_on_valid_report(
    valid_report: Report, sample_sources: list[SourceReference]
) -> None:
    """A valid report should pass all Gate 1 checks with zero critical or major defects."""
    verifier = DeterministicVerifier()
    result = verifier.verify(valid_report, sample_sources)

    assert result.passed is True
    assert result.schema_valid is True
    assert result.citations_valid is True
    assert result.passages_matched is True
    assert result.critical_count == 0
    assert result.major_count == 0
    assert result.total_claims == 4
    assert result.total_citations == 4


def test_normalized_quotation_matching(sample_sources: list[SourceReference]) -> None:
    """Passages with typographic quotes and extra whitespace should match as 'normalized'."""
    src = sample_sources[0]
    # Source contains: 'quantum annealing demonstrated a 3.2x speedup over classical simulated annealing'
    # We introduce typographic curly quotes and irregular whitespace:
    quoted_with_curly_quotes = (
        "“quantum   annealing  demonstrated   a 3.2x\n\tspeedup   over   classical”"
    )

    claim = Claim(
        claim_id="c_norm_1",
        section_id="sec_1",
        text="Speedup demonstrated.",
        citations=[
            Citation(
                citation_id="cite_norm_1",
                source_id=src.source_id,
                chunk_id=src.chunk_id,
                citation_type=CitationType.QUOTATION,
                quoted_passage=quoted_with_curly_quotes,
            )
        ],
    )
    report = Report(
        topic="Testing Normalization",
        title="Norm Test",
        executive_summary=ReportSection(
            section_id="sec_exec", title="Exec", content="Exec prose", claims=[]
        ),
        sections=[
            ReportSection(
                section_id="sec_1",
                title="Body",
                content="Body prose [cite_norm_1]",
                claims=[claim],
            )
        ],
        conclusion=ReportSection(
            section_id="sec_concl", title="Concl", content="Concl prose", claims=[]
        ),
    )

    verifier = DeterministicVerifier()
    result = verifier.verify(report, [src])

    assert result.passed is True
    assert result.passages_matched is True
    assert len(result.passage_matches) == 1
    match = result.passage_matches[0]
    assert match.is_match is True
    assert match.match_type == "normalized"


def test_quotation_missing_from_source_fails_strictly(
    valid_report: Report, sample_sources: list[SourceReference]
) -> None:
    """Quotation not found in the referenced source chunk must strictly fail as CRITICAL."""
    # Modify a citation to contain a fabricated quote
    valid_report.sections[0].claims[0].citations[
        0
    ].quoted_passage = "this sentence does not exist anywhere in the source document"

    verifier = DeterministicVerifier()
    result = verifier.verify(valid_report, sample_sources)

    assert result.passed is False
    assert result.passages_matched is False
    assert result.critical_count >= 1

    missing_issue = next(
        i
        for i in result.issues
        if i.category == DefectCategory.FACTUAL_GROUNDING and i.severity == DefectSeverity.CRITICAL
    )
    assert "does not exist in referenced chunk" in missing_issue.message


def test_strict_no_fuzzy_80_percent_pass(sample_sources: list[SourceReference]) -> None:
    """A passage with partial token overlap (>80%) but mutated words must NOT pass as quotation."""
    src = sample_sources[0]
    # Real text: "quantum annealing demonstrated a 3.2x speedup over classical simulated annealing"
    # Mutated text (paraphrase represented as quotation):
    mutated_passage = (
        "quantum annealing proved a 3.2x speed increase over traditional simulated annealing"
    )

    claim = Claim(
        claim_id="c_fuzzy_1",
        section_id="sec_1",
        text="Speedup assertion.",
        citations=[
            Citation(
                citation_id="cite_fuzzy_1",
                source_id=src.source_id,
                chunk_id=src.chunk_id,
                citation_type=CitationType.QUOTATION,
                quoted_passage=mutated_passage,
            )
        ],
    )
    report = Report(
        topic="Fuzzy Check",
        title="Fuzzy Test",
        executive_summary=ReportSection(
            section_id="sec_exec", title="Exec", content="Exec prose", claims=[]
        ),
        sections=[
            ReportSection(
                section_id="sec_1",
                title="Body",
                content="Body prose [cite_fuzzy_1]",
                claims=[claim],
            )
        ],
        conclusion=ReportSection(
            section_id="sec_concl", title="Concl", content="Concl prose", claims=[]
        ),
    )

    verifier = DeterministicVerifier()
    result = verifier.verify(report, [src])

    assert result.passed is False
    assert result.passages_matched is False
    assert any(
        i.severity == DefectSeverity.CRITICAL and "does not exist in referenced chunk" in i.message
        for i in result.issues
    )


def test_unknown_source_chunk_fails_critical(
    valid_report: Report, sample_sources: list[SourceReference]
) -> None:
    """Citation referencing a chunk ID not in the application source pool must fail as CRITICAL."""
    # Point citation to non-existent chunk
    valid_report.sections[0].claims[0].citations[0].chunk_id = "chunk_phantom_999"

    verifier = DeterministicVerifier()
    result = verifier.verify(valid_report, sample_sources)

    assert result.passed is False
    assert result.citations_valid is False
    assert any(
        i.severity == DefectSeverity.CRITICAL and "unknown source/chunk pair" in i.message
        for i in result.issues
    )


def test_uncited_factual_claim_fails_major(
    valid_report: Report, sample_sources: list[SourceReference]
) -> None:
    """A factual claim with no citations must generate a MAJOR defect."""
    # Remove citations from a factual claim
    valid_report.sections[0].claims[0].citations = []

    verifier = DeterministicVerifier()
    result = verifier.verify(valid_report, sample_sources)

    assert result.passed is False
    assert result.major_count >= 1
    assert any(
        i.severity == DefectSeverity.MAJOR and "lacks supporting citations" in i.message
        for i in result.issues
    )


def test_paraphrase_citation_does_not_fail_quotation_substring(
    valid_report: Report, sample_sources: list[SourceReference]
) -> None:
    """Citations of type PARAPHRASE without quotes should not fail Gate 1 substring matching."""
    # The conclusion claim in valid_report is PARAPHRASE with quoted_passage=None
    concl_cite = valid_report.conclusion.claims[0].citations[0]
    assert concl_cite.citation_type == CitationType.PARAPHRASE
    assert concl_cite.quoted_passage is None

    verifier = DeterministicVerifier()
    result = verifier.verify(valid_report, sample_sources)

    assert result.passed is True
    # Verify no passage match defect was raised for the paraphrase
    assert not any(i.citation_id == concl_cite.citation_id for i in result.issues)


def test_source_input_formats(valid_report: Report, sample_sources: list[SourceReference]) -> None:
    """Verifier should accept sources as list, tuple-keyed dict, or string-keyed dict."""
    verifier = DeterministicVerifier()

    # Format 1: List
    res_list = verifier.verify(valid_report, sample_sources)
    assert res_list.passed is True

    # Format 2: Dict keyed by (source_id, chunk_id)
    tuple_dict = {(s.source_id, s.chunk_id): s for s in sample_sources}
    res_tuple = verifier.verify(valid_report, tuple_dict)
    assert res_tuple.passed is True

    # Format 3: Dict keyed by chunk_id
    str_dict = {s.chunk_id: s for s in sample_sources}
    res_str = verifier.verify(valid_report, str_dict)
    assert res_str.passed is True
