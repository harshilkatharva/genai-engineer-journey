"""Shared test fixtures for module-19-reflection test suite."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import pytest

from self_improving_report_writer.models.report import (
    Citation,
    CitationType,
    Claim,
    Report,
    ReportSection,
    SourceReference,
)


def check_integration_gate(
    keywords: Mapping[str, Any] | set[str], integration_env: str | None
) -> None:
    """Skip test if it is marked with integration and INTEGRATION_TEST != '1'."""
    if "integration" in keywords and integration_env != "1":
        pytest.skip(
            "Integration tests are disabled. Set INTEGRATION_TEST=1 in .env or environment to run them."
        )


@pytest.fixture(autouse=True)
def require_integration_test_flag(request: pytest.FixtureRequest) -> None:
    """Gate tests marked with @pytest.mark.integration behind INTEGRATION_TEST=1."""
    check_integration_gate(request.keywords, os.getenv("INTEGRATION_TEST"))


@pytest.fixture
def sample_sources() -> list[SourceReference]:
    """Sample immutable source references in the application-controlled pool."""
    return [
        SourceReference(
            source_id="src_quantum_01",
            chunk_id="chunk_q01",
            title="Quantum Optimization Benchmarks 2026",
            content=(
                "In extensive benchmarks on logistics routing graphs, quantum annealing demonstrated "
                "a 3.2x speedup over classical simulated annealing on 500-node instances. However, "
                "on smaller graphs with fewer than 100 nodes, classical branch-and-bound remained faster."
            ),
            score=0.94,
            metadata={"domain": "quantum_computing", "year": 2026},
        ),
        SourceReference(
            source_id="src_quantum_02",
            chunk_id="chunk_q02",
            title="Gate-Model vs Annealing Review",
            content=(
                "While gate-model processors suffer from high error rates in NISQ devices, "
                "analog quantum annealers operate with 5,000+ qubits with manageable decoherence times."
            ),
            score=0.89,
            metadata={"domain": "quantum_computing", "year": 2025},
        ),
    ]


@pytest.fixture
def sample_source_map(
    sample_sources: list[SourceReference],
) -> dict[tuple[str, str], SourceReference]:
    """Lookup map for sample sources keyed by (source_id, chunk_id)."""
    return {(s.source_id, s.chunk_id): s for s in sample_sources}


@pytest.fixture
def valid_report() -> Report:
    """A structurally sound Report with valid citations matching sample_sources."""
    exec_claim = Claim(
        claim_id="claim_exec_1",
        section_id="sec_exec",
        text="Quantum annealing achieved a 3.2x speedup over simulated annealing on large graphs.",
        is_factual=True,
        citations=[
            Citation(
                citation_id="cite_exec_1",
                source_id="src_quantum_01",
                chunk_id="chunk_q01",
                citation_type=CitationType.QUOTATION,
                quoted_passage=(
                    "quantum annealing demonstrated a 3.2x speedup over classical simulated "
                    "annealing on 500-node instances"
                ),
            )
        ],
    )
    exec_sec = ReportSection(
        section_id="sec_exec",
        title="Executive Summary",
        content="This report analyzes quantum annealing performance [cite_exec_1].",
        claims=[exec_claim],
    )

    finding_claim_1 = Claim(
        claim_id="claim_find_1",
        section_id="sec_findings",
        text="On small graphs with under 100 nodes, classical branch-and-bound remained faster.",
        is_factual=True,
        citations=[
            Citation(
                citation_id="cite_find_1",
                source_id="src_quantum_01",
                chunk_id="chunk_q01",
                citation_type=CitationType.QUOTATION,
                quoted_passage="on smaller graphs with fewer than 100 nodes, classical branch-and-bound remained faster",
            )
        ],
    )
    finding_claim_2 = Claim(
        claim_id="claim_find_2",
        section_id="sec_findings",
        text="Analog quantum annealers can deploy over 5,000 qubits.",
        is_factual=True,
        citations=[
            Citation(
                citation_id="cite_find_2",
                source_id="src_quantum_02",
                chunk_id="chunk_q02",
                citation_type=CitationType.QUOTATION,
                quoted_passage="analog quantum annealers operate with 5,000+ qubits with manageable decoherence times",
            )
        ],
    )
    findings_sec = ReportSection(
        section_id="sec_findings",
        title="Detailed Findings",
        content=(
            "Benchmarking indicates clear operational thresholds [cite_find_1]. "
            "Hardware capacity continues to expand significantly [cite_find_2]."
        ),
        claims=[finding_claim_1, finding_claim_2],
    )

    concl_claim = Claim(
        claim_id="claim_concl_1",
        section_id="sec_concl",
        text="Classical solvers remain superior for small problem instances.",
        is_factual=True,
        citations=[
            Citation(
                citation_id="cite_concl_1",
                source_id="src_quantum_01",
                chunk_id="chunk_q01",
                citation_type=CitationType.PARAPHRASE,
            )
        ],
    )
    concl_sec = ReportSection(
        section_id="sec_concl",
        title="Conclusion",
        content="In conclusion, hybrid approaches leverage the best of both paradigms [cite_concl_1].",
        claims=[concl_claim],
    )

    return Report(
        topic="Quantum Annealing vs Classical Solvers in Logistics",
        title="Quantum Annealing in Combinatorial Logistics Optimization",
        executive_summary=exec_sec,
        sections=[findings_sec],
        conclusion=concl_sec,
        limitations=["Benchmarks limited to 500-node synthetic graphs."],
        version=1,
    )
