"""Deterministic verifier enforcing Gate 1 structural and citation invariants."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..models.report import CitationType, Report, SourceReference
from ..models.verification import (
    DefectCategory,
    DefectSeverity,
    DeterministicVerificationReport,
    PassageMatchResult,
    VerificationIssue,
)


def normalize_for_matching(text: str) -> str:
    """Normalize text for deterministic substring comparison.

    Lowercases text, collapses consecutive whitespace, normalizes typographic
    quotes and apostrophes, and strips non-breaking spaces.
    """
    if not text:
        return ""
    normalized = (
        text.lower()
        .replace("\xa0", " ")
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("`", "'")
    )
    return " ".join(normalized.split())


class DeterministicVerifier:
    """Gate 1 verifier: executes pure Python structural, pointer, and quotation checks."""

    def verify(
        self,
        report: Report,
        retrieved_sources: (
            dict[tuple[str, str], SourceReference]
            | dict[str, SourceReference]
            | Sequence[SourceReference]
        ),
    ) -> DeterministicVerificationReport:
        """Verify report against structural rules and application-controlled sources.

        Args:
            report: The Report instance to verify.
            retrieved_sources: The authoritative application-controlled source pool.
                Can be a dict keyed by (source_id, chunk_id), a dict keyed by chunk_id,
                or a sequence of SourceReference instances.

        Returns:
            DeterministicVerificationReport containing all findings and pass/fail verdict.
        """
        source_map = self._build_source_map(retrieved_sources)
        issues: list[VerificationIssue] = []
        passage_matches: list[PassageMatchResult] = []

        schema_valid = True
        citations_valid = True
        passages_matched = True

        all_sections = [report.executive_summary, *report.sections, report.conclusion]
        total_claims = 0
        total_citations = 0
        issue_counter = 1

        # 1. Structural Section Invariants
        if not report.sections:
            issues.append(
                VerificationIssue(
                    issue_id=f"DEFECT-{issue_counter:03d}",
                    category=DefectCategory.SCHEMA,
                    severity=DefectSeverity.CRITICAL,
                    message="Report must contain at least one body section in 'sections'.",
                )
            )
            issue_counter += 1
            schema_valid = False

        # 2. Iterate Sections, Claims, and Citations
        for sec in all_sections:
            sec_citation_ids: set[str] = set()

            for claim in sec.claims:
                total_claims += 1

                # Claim-Section binding check
                if claim.section_id != sec.section_id:
                    issues.append(
                        VerificationIssue(
                            issue_id=f"DEFECT-{issue_counter:03d}",
                            category=DefectCategory.SCHEMA,
                            severity=DefectSeverity.CRITICAL,
                            message=(
                                f"Claim '{claim.claim_id}' section_id='{claim.section_id}' "
                                f"does not match parent section '{sec.section_id}'."
                            ),
                            section_id=sec.section_id,
                            claim_id=claim.claim_id,
                        )
                    )
                    issue_counter += 1
                    schema_valid = False

                # Empty claim check
                if not claim.text.strip():
                    issues.append(
                        VerificationIssue(
                            issue_id=f"DEFECT-{issue_counter:03d}",
                            category=DefectCategory.SCHEMA,
                            severity=DefectSeverity.CRITICAL,
                            message=f"Claim '{claim.claim_id}' has empty text.",
                            section_id=sec.section_id,
                            claim_id=claim.claim_id,
                        )
                    )
                    issue_counter += 1
                    schema_valid = False

                # Factual claim citation presence check
                if claim.is_factual and not claim.citations:
                    issues.append(
                        VerificationIssue(
                            issue_id=f"DEFECT-{issue_counter:03d}",
                            category=DefectCategory.CITATION_INTEGRITY,
                            severity=DefectSeverity.MAJOR,
                            message=f"Factual claim '{claim.claim_id}' lacks supporting citations.",
                            section_id=sec.section_id,
                            claim_id=claim.claim_id,
                        )
                    )
                    issue_counter += 1
                    citations_valid = False

                for cite in claim.citations:
                    total_citations += 1
                    sec_citation_ids.add(cite.citation_id)

                    source_key = (cite.source_id, cite.chunk_id)
                    source_chunk = source_map.get(source_key)

                    # Source existence check in application-controlled pool
                    if source_chunk is None:
                        issues.append(
                            VerificationIssue(
                                issue_id=f"DEFECT-{issue_counter:03d}",
                                category=DefectCategory.CITATION_INTEGRITY,
                                severity=DefectSeverity.CRITICAL,
                                message=(
                                    f"Citation '{cite.citation_id}' references unknown "
                                    f"source/chunk pair ({cite.source_id}, {cite.chunk_id}) "
                                    "not present in application-controlled source pool."
                                ),
                                section_id=sec.section_id,
                                claim_id=claim.claim_id,
                                citation_id=cite.citation_id,
                                details={
                                    "referenced_source_id": cite.source_id,
                                    "referenced_chunk_id": cite.chunk_id,
                                },
                            )
                        )
                        issue_counter += 1
                        citations_valid = False
                        continue

                    # Exact / Normalized Quotation matching
                    if cite.citation_type == CitationType.QUOTATION:
                        if not cite.quoted_passage or not cite.quoted_passage.strip():
                            issues.append(
                                VerificationIssue(
                                    issue_id=f"DEFECT-{issue_counter:03d}",
                                    category=DefectCategory.CITATION_INTEGRITY,
                                    severity=DefectSeverity.CRITICAL,
                                    message=(
                                        f"Citation '{cite.citation_id}' is marked as QUOTATION "
                                        "but has empty quoted_passage."
                                    ),
                                    section_id=sec.section_id,
                                    claim_id=claim.claim_id,
                                    citation_id=cite.citation_id,
                                )
                            )
                            issue_counter += 1
                            citations_valid = False
                            continue

                        quoted = cite.quoted_passage
                        source_text = source_chunk.content

                        norm_quoted = normalize_for_matching(quoted)
                        norm_source = normalize_for_matching(source_text)
                        stripped_quoted = norm_quoted.strip("\"'")

                        if quoted in source_text:
                            passage_matches.append(
                                PassageMatchResult(
                                    citation_id=cite.citation_id,
                                    source_id=cite.source_id,
                                    chunk_id=cite.chunk_id,
                                    is_match=True,
                                    match_type="exact",
                                    quoted_passage=quoted,
                                    normalized_passage=norm_quoted,
                                    normalized_source_chunk=norm_source,
                                )
                            )
                        elif norm_quoted in norm_source or (
                            stripped_quoted and stripped_quoted in norm_source
                        ):
                            passage_matches.append(
                                PassageMatchResult(
                                    citation_id=cite.citation_id,
                                    source_id=cite.source_id,
                                    chunk_id=cite.chunk_id,
                                    is_match=True,
                                    match_type="normalized",
                                    quoted_passage=quoted,
                                    normalized_passage=norm_quoted,
                                    normalized_source_chunk=norm_source,
                                )
                            )
                        else:
                            # Strict match failure: no fuzzy 80% passing!
                            passage_matches.append(
                                PassageMatchResult(
                                    citation_id=cite.citation_id,
                                    source_id=cite.source_id,
                                    chunk_id=cite.chunk_id,
                                    is_match=False,
                                    match_type="missing",
                                    quoted_passage=quoted,
                                    normalized_passage=norm_quoted,
                                    normalized_source_chunk=norm_source,
                                )
                            )
                            issues.append(
                                VerificationIssue(
                                    issue_id=f"DEFECT-{issue_counter:03d}",
                                    category=DefectCategory.FACTUAL_GROUNDING,
                                    severity=DefectSeverity.CRITICAL,
                                    message=(
                                        f"Quotation in citation '{cite.citation_id}' does not "
                                        f"exist in referenced chunk '{cite.chunk_id}'."
                                    ),
                                    section_id=sec.section_id,
                                    claim_id=claim.claim_id,
                                    citation_id=cite.citation_id,
                                    details={
                                        "quoted_passage": quoted,
                                        "referenced_chunk_id": cite.chunk_id,
                                    },
                                )
                            )
                            issue_counter += 1
                            passages_matched = False

            # 3. Check for unreferenced citations in section prose
            for cite_id in sec_citation_ids:
                if cite_id not in sec.content and not self._citation_marker_in_prose(
                    cite_id, sec.content
                ):
                    issues.append(
                        VerificationIssue(
                            issue_id=f"DEFECT-{issue_counter:03d}",
                            category=DefectCategory.ORGANIZATION_CLARITY,
                            severity=DefectSeverity.MINOR,
                            message=(
                                f"Citation '{cite_id}' is defined on claim in section "
                                f"'{sec.section_id}' but its identifier does not appear in section prose."
                            ),
                            section_id=sec.section_id,
                            citation_id=cite_id,
                        )
                    )
                    issue_counter += 1

        crit_count = sum(1 for i in issues if i.severity == DefectSeverity.CRITICAL)
        maj_count = sum(1 for i in issues if i.severity == DefectSeverity.MAJOR)
        min_count = sum(1 for i in issues if i.severity == DefectSeverity.MINOR)

        passed = crit_count == 0 and maj_count == 0 and schema_valid and citations_valid

        return DeterministicVerificationReport(
            passed=passed,
            schema_valid=schema_valid,
            citations_valid=citations_valid,
            passages_matched=passages_matched,
            total_claims=total_claims,
            total_citations=total_citations,
            passage_matches=passage_matches,
            issues=issues,
            critical_count=crit_count,
            major_count=maj_count,
            minor_count=min_count,
        )

    def _citation_marker_in_prose(self, citation_id: str, content: str) -> bool:
        """Check if citation ID appears with standard bracket notation like [cite_1]."""
        pattern = r"\[\s*" + re.escape(citation_id) + r"\s*\]"
        return bool(re.search(pattern, content))

    def _build_source_map(
        self,
        retrieved_sources: (
            dict[tuple[str, str], SourceReference]
            | dict[str, SourceReference]
            | Sequence[SourceReference]
        ),
    ) -> dict[tuple[str, str], SourceReference]:
        """Normalize retrieved source collection into a (source_id, chunk_id) lookup dict."""
        source_map: dict[tuple[str, str], SourceReference] = {}

        if isinstance(retrieved_sources, dict):
            for k, v in retrieved_sources.items():
                if isinstance(k, tuple) and len(k) == 2:
                    source_map[(str(k[0]), str(k[1]))] = v
                else:
                    source_map[(v.source_id, v.chunk_id)] = v
        else:
            for s in retrieved_sources:
                source_map[(s.source_id, s.chunk_id)] = s

        return source_map
