"""The 10 curated benchmark goals for Module 18 Plan-and-Execute Assistant."""

from __future__ import annotations

from genai_core.evaluation import EvalDataset
from genai_core.models.evaluation_model import (
    EvalSample,
    ExpectedAnswer,
    ExpectedEvidence,
)


def get_10_benchmark_goals() -> list[EvalSample]:
    """Generate the 10 curated benchmark evaluation scenarios covering all required categories.

    Categories:
    - Research: multi-source information gathering, synthesis, comparison matrices.
    - Operations: sequential CRM/ticketing actions, status transitions, gated operations.
    - Resilience: ambiguous goals, verification failures triggering dynamic replanning.
    - Mixed: combined database, knowledge-base, and long-term memory reasoning.
    """
    return [
        # G-01: Research (Parallel branches joining at synthesis)
        EvalSample(
            id="G-01",
            query="Compare HNSW and IVFFlat vector indexing algorithms in terms of latency, build time, and recall.",
            expected_tools=["search_web", "search_knowledge_base"],
            expected_evidence=ExpectedEvidence(required=["HNSW", "IVFFlat"]),
            expected_answer=ExpectedAnswer(
                required=["HNSW", "IVFFlat", "recall", "latency"],
                reference=(
                    "HNSW provides lower query latency and higher recall at the expense of longer index "
                    "build times and higher memory footprint. IVFFlat offers faster index build times and "
                    "lower memory usage but requires clustering training and trades off recall at high query loads."
                ),
            ),
            metadata={
                "category": "research",
                "user_id": "user_researcher",
                "test_focus": "Parallel independent research branches joining at synthesis",
            },
        ),
        # G-02: Operations (Strict sequential order dependencies)
        EvalSample(
            id="G-02",
            query="Investigate customer account C-101, inspect recent billing tickets, verify refund policy, and draft customer response.",
            expected_tools=["lookup_customer_account", "ticket_status", "search_knowledge_base"],
            expected_evidence=ExpectedEvidence(required=["C-101", "refund"]),
            expected_answer=ExpectedAnswer(
                required=["C-101", "refund", "ticket"],
                reference=(
                    "Customer C-101 is in the Gold tier. Recent billing ticket T-101 was inspected. "
                    "Under company refund policy, Gold tier accounts qualify for prorated refunds within 30 days. "
                    "Customer response drafted detailing the approved adjustment."
                ),
            ),
            metadata={
                "category": "operations",
                "user_id": "C-101",
                "test_focus": "Strict sequential order dependencies across DB and KB",
            },
        ),
        # G-03: Operations (Gated operational sequence requiring inspection before write)
        EvalSample(
            id="G-03",
            query="Resolve high-priority ticket T-102 for customer C-102 and update customer account tier if eligible.",
            expected_tools=["ticket_status", "update_ticket_status", "lookup_customer_account"],
            expected_evidence=ExpectedEvidence(required=["T-102", "resolved"]),
            expected_answer=ExpectedAnswer(
                required=["T-102", "resolved"],
                reference=(
                    "High-priority ticket T-102 inspected and updated to RESOLVED. "
                    "Associated customer account C-102 eligibility verified for tier upgrade."
                ),
            ),
            metadata={
                "category": "operations",
                "user_id": "C-102",
                "test_focus": "Gated sequence requiring inspection before status write",
            },
        ),
        # G-04: Research (Multi-query KB retrieval combined with external web search)
        EvalSample(
            id="G-04",
            query="Perform security audit comparing internal knowledge-base guidelines with public OWASP Top 10 web vulnerabilities.",
            expected_tools=["search_knowledge_base", "search_web"],
            expected_evidence=ExpectedEvidence(required=["OWASP", "vulnerability"]),
            expected_answer=ExpectedAnswer(
                required=["OWASP", "vulnerability", "security"],
                reference=(
                    "Internal security guidelines require input validation and parameterized queries. "
                    "Cross-referenced against OWASP Top 10 vulnerabilities (Injection, Broken Access Control, "
                    "and Cryptographic Failures), internal practices meet compliance with recommendations for automated SAST scanning."
                ),
            ),
            metadata={
                "category": "research",
                "user_id": "user_sec_auditor",
                "test_focus": "Multi-query KB retrieval combined with external web search",
            },
        ),
        # G-05: Mixed (Multi-tier reasoning combining database and long-term memory)
        EvalSample(
            id="G-05",
            query="Diagnose customer churn risk for customer C-104 using ticket history, stored memory preferences, and account tier.",
            expected_tools=["lookup_customer_account", "ticket_status"],
            expected_evidence=ExpectedEvidence(required=["C-104", "churn"]),
            expected_answer=ExpectedAnswer(
                required=["C-104", "churn", "risk"],
                reference=(
                    "Customer C-104 exhibits elevated churn risk due to repeated billing escalation tickets "
                    "and recorded preference for rapid SLA turnaround. Recommended proactive retention outreach."
                ),
            ),
            metadata={
                "category": "mixed",
                "user_id": "C-104",
                "test_focus": "Multi-tier reasoning combining database and long-term memory",
            },
        ),
        # G-06: Resilience (Dynamic replanning triggered after initial verification failure)
        EvalSample(
            id="G-06",
            query="Retrieve documentation for an ambiguous query requiring dynamic replanning after initial verification failure.",
            expected_tools=["search_knowledge_base", "search_web"],
            expected_evidence=ExpectedEvidence(required=["documentation", "retrieved"]),
            expected_answer=ExpectedAnswer(
                required=["documentation", "retrieved"],
                reference=(
                    "Initial targeted retrieval failed to satisfy precision criteria. "
                    "Replanning engine generated broader contextual query and alternative search terms, "
                    "successfully recovering the required operational documentation."
                ),
            ),
            metadata={
                "category": "resilience",
                "user_id": "user_resilience_test",
                "test_focus": "Verifier detecting insufficient output and triggering dynamic replan",
            },
        ),
        # G-07: Operations (3 independent parallel fetch steps merging into 1 synthesis step)
        EvalSample(
            id="G-07",
            query="Batch status and resolution audit across tickets T-101, T-103, and T-105 with a single consolidated executive brief.",
            expected_tools=["ticket_status"],
            expected_evidence=ExpectedEvidence(required=["T-101", "T-103", "T-105"]),
            expected_answer=ExpectedAnswer(
                required=["T-101", "T-103", "T-105"],
                reference=(
                    "Consolidated status audit for tickets T-101, T-103, and T-105 compiled in parallel. "
                    "Executive brief summarizes open vs resolved states and resolution bottlenecks."
                ),
            ),
            metadata={
                "category": "operations",
                "user_id": "lead_support_mgr",
                "test_focus": "3 independent parallel fetch steps merging into 1 synthesis step",
            },
        ),
        # G-08: Research (Multi-source web search with structured table deliverable)
        EvalSample(
            id="G-08",
            query="Analyze pricing models and SLA terms for three cloud vector databases from web sources with a markdown comparison matrix.",
            expected_tools=["search_web"],
            expected_evidence=ExpectedEvidence(required=["pricing", "vector"]),
            expected_answer=ExpectedAnswer(
                required=["pricing", "SLA", "vector"],
                reference=(
                    "Markdown comparison matrix analyzing Pinecone, Qdrant Cloud, and Weaviate Cloud "
                    "across hourly compute, storage costs, and enterprise SLA availability."
                ),
            ),
            metadata={
                "category": "research",
                "user_id": "user_architect",
                "test_focus": "Multi-source web search with structured table deliverable",
            },
        ),
        # G-09: Operations (Verification of prerequisites prior to irreversible action)
        EvalSample(
            id="G-09",
            query="Prepare refund execution for customer C-102: verify customer eligibility, calculate prorated refund amount, and record audit log.",
            expected_tools=[
                "lookup_customer_account",
                "search_knowledge_base",
                "save_longterm_memory",
            ],
            expected_evidence=ExpectedEvidence(required=["C-102", "eligibility", "refund"]),
            expected_answer=ExpectedAnswer(
                required=["C-102", "eligibility", "refund", "amount"],
                reference=(
                    "Customer C-102 eligibility verified against active subscription period. "
                    "Prorated refund amount calculated and audit log entry staged."
                ),
            ),
            metadata={
                "category": "operations",
                "user_id": "C-102",
                "test_focus": "Verification of eligibility prerequisites prior to irreversible action",
            },
        ),
        # G-10: Mixed (Multi-hop reasoning where subsequent step depends on initial discovery)
        EvalSample(
            id="G-10",
            query="Root-cause analysis for system incident on ticket T-103: search knowledge-base guidelines, query incident ticket, and draft postmortem.",
            expected_tools=["search_knowledge_base", "ticket_status"],
            expected_evidence=ExpectedEvidence(required=["T-103", "incident", "root-cause"]),
            expected_answer=ExpectedAnswer(
                required=["T-103", "incident", "root-cause", "postmortem"],
                reference=(
                    "Discovered kernel crash panic escalation in documentation correlating with incident ticket T-103. "
                    "Drafted incident postmortem detailing root cause, timeline, and mitigation steps."
                ),
            ),
            metadata={
                "category": "mixed",
                "user_id": "C-102",
                "test_focus": "Multi-hop reasoning where subsequent step depends on initial discovery",
            },
        ),
    ]


def create_10_goals_dataset(name: str = "module_18_10_goals_benchmark") -> EvalDataset:
    """Construct an EvalDataset instance containing all 10 benchmark goals."""
    return EvalDataset(
        samples=get_10_benchmark_goals(),
        metadata={
            "name": name,
            "description": "10-goal comparative evaluation benchmark for Planned vs. Reactive architectures",
            "module": 18,
            "version": "1.0.0",
        },
    )


BENCHMARK_GOALS: list[EvalSample] = get_10_benchmark_goals()
