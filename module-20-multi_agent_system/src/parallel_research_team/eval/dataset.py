"""Curated 15-question benchmark dataset comparing Single Agent vs Multi-Agent systems."""

from __future__ import annotations

from .models import EvaluationCategory, EvaluationQuestion, ExpectedWinner

BENCHMARK_15_QUESTIONS: list[EvaluationQuestion] = [
    # =========================================================================
    # Category 1: Simple / Direct Factual Retrieval (Single-Agent should win)
    # =========================================================================
    EvaluationQuestion(
        id="q01_clarissa_recipient",
        query="Who is the primary recipient of Clarissa Harlowe's letters in Volume 1?",
        category=EvaluationCategory.SIMPLE_FACTUAL,
        expected_winner=ExpectedWinner.SINGLE_AGENT,
        single_agent_hypothesis="A single targeted search retrieves Miss Anna Howe in 1 tool step (<5s, low cost).",
        multi_agent_hypothesis="Decomposition and multi-agent coordination add unnecessary token and latency overhead with no quality gain.",
        ground_truth_reference="Miss Anna Howe is the primary confidante and recipient of Clarissa Harlowe's letters in Volume 1 of Samuel Richardson's novel Clarissa.",
        rubric_focus=["factuality", "brevity", "cost_efficiency"],
    ),
    EvaluationQuestion(
        id="q02_ivfflat_concept",
        query="What does the abbreviation IVFFlat stand for in vector indexing, and what is its primary cluster-based partitioning concept?",
        category=EvaluationCategory.SIMPLE_FACTUAL,
        expected_winner=ExpectedWinner.SINGLE_AGENT,
        single_agent_hypothesis="Direct conceptual lookup; single agent retrieves Inverted File Flat and Voronoi cells in 1 step.",
        multi_agent_hypothesis="Multi-agent fan-out provides zero benefit for a single well-defined algorithmic concept.",
        ground_truth_reference="IVFFlat stands for Inverted File Flat. It partitions the high-dimensional vector space into Voronoi cells using k-means clustering. At query time, only vectors in the nearest centroids (nprobe) are scanned.",
        rubric_focus=["factuality", "technical_accuracy"],
    ),
    EvaluationQuestion(
        id="q03_tesla_patent_year",
        query="In what year was Nikola Tesla granted his foundational US patent for the alternating-current electro-magnetic motor (US Patent 381,968)?",
        category=EvaluationCategory.SIMPLE_FACTUAL,
        expected_winner=ExpectedWinner.SINGLE_AGENT,
        single_agent_hypothesis="Single date lookup; single agent completes rapidly in 1 step.",
        multi_agent_hypothesis="Orchestrator decomposition overhead drastically inflates latency and token cost for a single-fact lookup.",
        ground_truth_reference="Nikola Tesla was granted US Patent No. 381,968 for his Electro-Magnetic Motor on May 1, 1888 (filed October 12, 1887).",
        rubric_focus=["factuality", "exact_date"],
    ),
    EvaluationQuestion(
        id="q04_minilm_dimension",
        query="What is the standard vector embedding output dimensionality of the all-MiniLM-L6-v2 sentence-transformers model?",
        category=EvaluationCategory.SIMPLE_FACTUAL,
        expected_winner=ExpectedWinner.SINGLE_AGENT,
        single_agent_hypothesis="Direct parameter query; single agent answers 384 dimensions immediately.",
        multi_agent_hypothesis="Multi-agent systems provide no leverage on atomic dimension queries.",
        ground_truth_reference="The all-MiniLM-L6-v2 sentence-transformers embedding model produces dense vector representations of exactly 384 dimensions.",
        rubric_focus=["factuality", "exact_number"],
    ),
    # =========================================================================
    # Category 2: Focused Linear Synthesis (Single-Agent should win or tie)
    # =========================================================================
    EvaluationQuestion(
        id="q05_pricing_calculation",
        query="Under standard Gemini 3.5 Flash Lite pricing ($0.075 per 1M input tokens and $0.30 per 1M output tokens), calculate the exact cost in USD for a research run that consumes 40,000 input tokens and 8,000 output tokens.",
        category=EvaluationCategory.LINEAR_SYNTHESIS,
        expected_winner=ExpectedWinner.SINGLE_AGENT,
        single_agent_hypothesis="Sequential linear calculation; single agent computes the exact $0.0054 USD without orchestration latency.",
        multi_agent_hypothesis="Splitting a single mathematical calculation across parallel workers risks calculation discrepancy or redundant work.",
        ground_truth_reference="Input cost = 40,000 * 0.075 / 1,000,000 = $0.0030 USD. Output cost = 8,000 * 0.30 / 1,000,000 = $0.0024 USD. Total cost = $0.0054 USD (or 0.54 cents).",
        rubric_focus=["mathematical_accuracy", "factuality"],
    ),
    EvaluationQuestion(
        id="q06_clarissa_family_motive",
        query="Which suitor is Clarissa Harlowe's family trying to force her to marry, and what specific financial and social reasons motivate their insistence?",
        category=EvaluationCategory.LINEAR_SYNTHESIS,
        expected_winner=ExpectedWinner.SINGLE_AGENT,
        single_agent_hypothesis="Focused literary inquiry; a single agent retrieves Mr. Solmes and the family's desire to unite estates in 1-2 tool calls.",
        multi_agent_hypothesis="Multi-agent decomposition does not provide tangible depth beyond what a single grounded search uncovers.",
        ground_truth_reference="Clarissa's family attempts to force her to marry Mr. Solmes. They favor him because of his extensive property and immense wealth, which would enrich the Harlowe family estate, despite his boorish character and Clarissa's intense abhorrence.",
        rubric_focus=["factuality", "comprehensiveness"],
    ),
    EvaluationQuestion(
        id="q07_hnsw_skiplist_heuristic",
        query="How does the Hierarchical Navigable Small World (HNSW) algorithm use a multi-layer skip-list structure to achieve logarithmic search complexity?",
        category=EvaluationCategory.LINEAR_SYNTHESIS,
        expected_winner=ExpectedWinner.SINGLE_AGENT,
        single_agent_hypothesis="Cohesive single-topic technical explanation; single agent explains hierarchical layers without coordination overhead.",
        multi_agent_hypothesis="Multi-agent decomposition adds unnecessary complexity to a unified algorithmic explanation.",
        ground_truth_reference="HNSW constructs a hierarchy of proximity graphs analogous to probabilistic skip-lists. Upper layers have fewer nodes with long-range edges for rapid coarse greedy routing; progressively lower layers become denser with shorter links until the ground layer (layer 0) where the nearest neighbor is precisely identified in O(log N) time.",
        rubric_focus=["technical_depth", "clarity"],
    ),
    # =========================================================================
    # Category 3: Multi-Faceted Comparative Research (Multi-Agent should win)
    # =========================================================================
    EvaluationQuestion(
        id="q08_hnsw_vs_ivfflat",
        query="Compare HNSW and IVFFlat vector database index structures across index build time, query search latency, memory consumption, and recall accuracy.",
        category=EvaluationCategory.MULTI_FACETED_COMPARISON,
        expected_winner=ExpectedWinner.MULTI_AGENT,
        single_agent_hypothesis="Single agent struggles to cover all four metrics in depth, often missing build time or memory overhead due to step limit.",
        multi_agent_hypothesis="Multi-agent decomposes the 4 dimensions across dedicated parallel worker briefs and synthesizes an authoritative trade-off matrix.",
        ground_truth_reference="HNSW: Slower build time, extremely low search latency, high memory usage (storing proximity graph edges), and high recall (>95%). IVFFlat: Fast build time (k-means clustering), moderate/higher latency depending on nprobe, low memory footprint (flat vector lists per cluster), and moderate recall subject to Voronoi boundary loss.",
        rubric_focus=["comprehensiveness", "comparative_matrix", "factuality"],
    ),
    EvaluationQuestion(
        id="q09_tesla_sync_vs_torque",
        query="What are the core operational differences between Nikola Tesla's synchronous motor and his induction torque motor regarding rotor slip, speed regulation, and torque production?",
        category=EvaluationCategory.MULTI_FACETED_COMPARISON,
        expected_winner=ExpectedWinner.MULTI_AGENT,
        single_agent_hypothesis="Single agent tends to conflate synchronous and induction principles or provide only a high-level summary.",
        multi_agent_hypothesis="Multi-agent dispatches specialized parallel briefs to investigate Tesla's synchronous lock-step patent vs induction slip requirements, synthesizing a rigorous comparison.",
        ground_truth_reference="Synchronous motor: Rotor locks into synchronism with the rotating stator field (zero slip) maintaining constant speed governed by supply frequency regardless of normal load. Torque/induction motor: Depends strictly on mechanical lag/slip (typically 2-5%) between rotor and stator field to induce currents and generate torque; if synchronous speed were reached, induction and torque would collapse to zero.",
        rubric_focus=["technical_depth", "factuality", "grounding"],
    ),
    EvaluationQuestion(
        id="q10_rag_eval_metrics",
        query="Compare the four primary RAG evaluation metrics—Context Precision, Context Recall, Faithfulness, and Answer Relevance—explaining their measurement methodology and failure modes.",
        category=EvaluationCategory.MULTI_FACETED_COMPARISON,
        expected_winner=ExpectedWinner.MULTI_AGENT,
        single_agent_hypothesis="Single agent hits step limit or summarizes loosely, failing to define measurement formulas and failure modes for all 4 metrics.",
        multi_agent_hypothesis="Multi-agent partitions retrieval metrics (Precision/Recall) from generation metrics (Faithfulness/Relevance), yielding comprehensive coverage.",
        ground_truth_reference="Context Precision measures whether retrieved chunks contain relevant signal ranked near the top. Context Recall measures whether all ground-truth facts are present in retrieved chunks. Faithfulness measures whether generated claims are strictly entailed by the context (hallucination detector). Answer Relevance measures whether the response directly addresses the user prompt without extraneous noise.",
        rubric_focus=["comprehensiveness", "metric_definitions", "failure_modes"],
    ),
    EvaluationQuestion(
        id="q11_pgvector_vs_dedicated_vdb",
        query="Compare PostgreSQL pgvector against dedicated vector databases (Pinecone, Qdrant, Milvus) for production enterprise systems on operational simplicity, horizontal scaling, hybrid relational filtering, and index performance.",
        category=EvaluationCategory.MULTI_FACETED_COMPARISON,
        expected_winner=ExpectedWinner.MULTI_AGENT,
        single_agent_hypothesis="Single agent provides generic pros/cons without deep operational insights into ACID vs distributed architecture.",
        multi_agent_hypothesis="Multi-agent splits research between relational ACID coexistence and distributed high-throughput search, producing an enterprise-grade architectural evaluation.",
        ground_truth_reference="pgvector excels in operational simplicity (no new database), single-transaction ACID compliance, and zero-latency relational filtering (JOINs with user tables). Dedicated vector databases (Qdrant, Pinecone, Milvus) excel in horizontal sharding, billions of vectors, advanced quantization (PQ, SQ), and sub-10ms search latencies under high QPS.",
        rubric_focus=["enterprise_depth", "architectural_tradeoffs", "comprehensiveness"],
    ),
    # =========================================================================
    # Category 4: Broad Cross-Disciplinary & Architectural Survey (Multi-Agent should win)
    # =========================================================================
    EvaluationQuestion(
        id="q12_production_genai_scope",
        query="What is the comprehensive scope of production Generative AI systems, covering offline data preparation, vector indexing, multi-agent orchestration, and post-generation reflection verification?",
        category=EvaluationCategory.ARCHITECTURAL_SURVEY,
        expected_winner=ExpectedWinner.MULTI_AGENT,
        single_agent_hypothesis="Single agent cannot cover all 4 architectural tiers within its bounded context and step limit, truncating details.",
        multi_agent_hypothesis="Multi-agent partitions the 4 tiers into parallel exploratory deep-dives, synthesizing a full-stack production reference architecture.",
        ground_truth_reference="Scope encompasses: (1) Data ingestion & chunking (sliding window, semantics, metadata), (2) Vector indexing (HNSW/IVFFlat, hybrid search, rerankers), (3) Multi-agent orchestration (bounded workers, budget isolation, shared state, failure shielding), and (4) Post-generation reflection & verification (deterministic structural gates, semantic entailment judges, revision loops).",
        rubric_focus=["comprehensiveness", "architectural_structure", "clarity"],
    ),
    EvaluationQuestion(
        id="q13_tesla_niagara_impact",
        query="Analyze the historical and technical progression of Nikola Tesla's 1888 polyphase alternating current inventions, the Westinghouse partnership, and their role in the 1895 Niagara Falls hydroelectric project.",
        category=EvaluationCategory.ARCHITECTURAL_SURVEY,
        expected_winner=ExpectedWinner.MULTI_AGENT,
        single_agent_hypothesis="Single agent produces a high-level summary that glosses over technical generator specifics or patent timeline details.",
        multi_agent_hypothesis="Multi-agent concurrently explores the 1888 patent disclosures, the War of the Currents legal battle, and the Niagara 2-phase powerhouse implementation.",
        ground_truth_reference="Nikola Tesla's 1888 polyphase motor and transformer patents solved long-distance AC power transmission. George Westinghouse licensed Tesla's patents in 1888. Despite Edison's DC advocacy, the success of the 1893 Chicago World's Fair led to the Niagara Falls Power Company contract, inaugurating the Adams Powerhouse in 1895 delivering 5,000-hp two-phase AC power to Buffalo over 26 miles, establishing AC as the global industrial standard.",
        rubric_focus=["historical_progression", "technical_accuracy", "comprehensiveness"],
    ),
    EvaluationQuestion(
        id="q14_agentic_architectures_tradeoffs",
        query="Analyze the architectural trade-offs between single-agent tool loops (ReAct), static DAG workflows (LangGraph), and dynamic multi-agent teams on cost, latency, failure isolation, and debugging complexity.",
        category=EvaluationCategory.ARCHITECTURAL_SURVEY,
        expected_winner=ExpectedWinner.MULTI_AGENT,
        single_agent_hypothesis="Single agent lacks the depth to systematically break down all 3 paradigms across all 4 operational axes.",
        multi_agent_hypothesis="Multi-agent assigns distinct worker briefs to investigate single-agent loops, static DAGs, and multi-agent teams, compiling a comprehensive matrix.",
        ground_truth_reference="ReAct single loops: Low initial cost, simple debugging, but high failure vulnerability (single failure collapses run) and poor parallelization. Static DAGs: Deterministic paths, predictable latency and cost, but brittle against dynamic branches. Dynamic multi-agent teams: Superior failure isolation and deep coverage via concurrent specialization, but higher token consumption, orchestration latency, and non-deterministic debugging.",
        rubric_focus=["comparative_matrix", "architectural_depth", "clarity"],
    ),
    EvaluationQuestion(
        id="q15_enterprise_llm_cache_guardrails",
        query="Examine enterprise production GenAI guardrail architectures, comparing semantic caching mechanisms, input sanitization/jailbreak detection, and output hallucination verification.",
        category=EvaluationCategory.ARCHITECTURAL_SURVEY,
        expected_winner=ExpectedWinner.MULTI_AGENT,
        single_agent_hypothesis="Single agent focuses on one or two guardrails (e.g. input prompt injection) and ignores semantic cache eviction or output entailment.",
        multi_agent_hypothesis="Multi-agent partitions cache architecture, input security, and output grounding verification across parallel audits.",
        ground_truth_reference="Semantic caching: Uses vector similarity to serve cached responses for semantically equivalent queries, slashing latency and cost, requiring strict similarity thresholding and TTL/eviction policies. Input sanitization: Detects prompt injections, jailbreaks, PII masking, and schema conformance before LLM execution. Output hallucination verification: Evaluates factual entailment between generated claims and retrieved source chunks (e.g. Module 19 Gates 1 & 2) before presentation.",
        rubric_focus=["security_depth", "comprehensiveness", "system_architecture"],
    ),
]


def get_15_benchmark_questions() -> list[EvaluationQuestion]:
    """Retrieve the full set of 15 benchmark evaluation questions."""
    return list(BENCHMARK_15_QUESTIONS)


def get_benchmark_question_by_id(question_id: str) -> EvaluationQuestion | None:
    """Retrieve a specific evaluation question by its ID."""
    for q in BENCHMARK_15_QUESTIONS:
        if q.id == question_id:
            return q
    return None
