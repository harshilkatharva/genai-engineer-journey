# Module 20: Parallel Research Team — Confirmed Business Logic

## 1. Domain Models, States & Schemas

### 1.1 Execution States & Enums

#### `WorkerStatus`
Governs the lifecycle of an individual research worker executing a brief:
- `PENDING`: Initial state upon brief generation. Awaiting dispatch by scheduler.
- `RUNNING`: Actively executing tool loops and LLM calls in parallel.
- `COMPLETED`: Worker completed all research objectives and produced validated findings.
- `PARTIAL`: Worker encountered a timeout or step limit, but produced salvageable partial findings.
- `FAILED`: Worker threw an unhandled exception or failed output validation completely.
- `TIMEOUT`: Worker exceeded configured per-worker wall-clock deadline.
- `SKIPPED`: Worker was bypassed (e.g. redundant brief or prompt classified as direct execution).

#### `RunStatus`
Governs the end-to-end Parallel Research Team execution lifecycle:
- `INITIALIZING`: Ingesting research question and recalling preferences/goals from memory.
- `DECOMPOSING`: Lead Agent evaluating effort scaling and generating typed worker briefs.
- `EXECUTING`: Dispatched to `ParallelResearchScheduler` for concurrent worker execution.
- `VALIDATING`: Validating worker outputs and building the authoritative source pool.
- `SYNTHESIZING`: Lead Agent compiling validated findings into a structured, cited report.
- `VERIFYING`: Running Module 19 citation and grounding verification gates on the final report.
- `COMPLETED`: Report synthesized, verified, and saved to memory and checkpoint.
- `PARTIALLY_COMPLETED`: Run succeeded with partial worker coverage (failed workers noted in report limitations).
- `FAILED`: Global run failure (e.g., global budget exhausted during decomposition, or all workers failed).
- `INTERRUPTED`: Run suspended or crashed before completion, ready for resume.

#### `EffortLevel`
Governs the decomposition strategy based on prompt complexity:
- `DIRECT`: Narrow, factual, or simple question. Dispatches 1 worker or executes a direct research path without parallel overhead.
- `PARALLEL_DECOMPOSED`: Broad, multi-faceted, or comparative question. Dispatches 2 to 5 parallel workers with disjoint briefs.

---

### 1.2 Core Data Schemas

```python
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


class BriefType(StrEnum):
    FACTUAL_RETRIEVAL = "factual_retrieval"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    EXPLORATORY_DEEP_DIVE = "exploratory_deep_dive"
    DIRECT_SYNTHESIS = "direct_synthesis"


class OutputFormatSpec(BaseModel):
    """JSON-serializable schema specification describing expected worker output format."""

    schema_name: str = Field(
        default="WorkerResearchOutput",
        description="Target Pydantic model name for the output",
    )
    required_fields: list[str] = Field(
        default_factory=lambda: ["findings_summary", "claims", "sources"],
        description="Mandatory JSON fields expected in the output",
    )
    guidelines: str = Field(
        default="Structured findings with claims and source citations",
        description="Format guidelines provided to the worker prompt",
    )


class WorkerBrief(BaseModel):
    """Typed research brief assigned to an independent research worker."""

    brief_id: str = Field(description="Unique brief identifier, e.g. 'brief_01'")
    title: str = Field(description="Short descriptive title of the assigned sub-topic")
    topic: str = Field(description="Overarching research topic")
    objective: str = Field(description="Specific research question or hypothesis to investigate")
    brief_type: BriefType = Field(default=BriefType.EXPLORATORY_DEEP_DIVE)
    worker_type: str = Field(
        default="default",
        description="Worker archetype name dispatched via WorkerRegistry factory",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Optional list of predecessor brief IDs required before this brief can run",
    )
    task_boundaries: list[str] = Field(
        description="Explicit boundaries: what is in-scope vs out-of-scope to prevent overlap"
    )
    expected_output_format: OutputFormatSpec = Field(
        default_factory=OutputFormatSpec,
        description="Typed, serializable schema reference for the worker deliverable",
    )
    suggested_tools: list[str] = Field(
        default_factory=lambda: ["search_web", "search_knowledge_base"],
        description="Tools recommended for this brief",
    )
    max_steps: int = Field(default=5, ge=1, le=10, description="Per-worker step ceiling")
    timeout_seconds: float = Field(
        default=45.0, gt=0.0, le=180.0, description="Per-worker timeout in seconds"
    )


class WorkerResearchOutput(BaseModel):
    """Structured deliverable produced by a research worker upon brief execution."""

    brief_id: str
    findings_summary: str = Field(description="Synthesis of gathered evidence for this brief")
    claims: list[dict[str, Any]] = Field(
        default_factory=list, description="Extracted factual claims with source links"
    )
    sources: list[dict[str, Any]] = Field(
        default_factory=list, description="Retrieved raw source chunks with source_id and chunk_id"
    )
    search_queries_used: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(
        default_factory=list, description="Gaps, unreachable sources, or uncertainties encountered"
    )


class WorkerExecutionResult(BaseModel):
    """Full execution record for a worker attempt."""

    brief_id: str
    status: WorkerStatus
    output: WorkerResearchOutput | None = None
    tokens_used: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    duration_ms: float = Field(default=0.0, ge=0.0)
    error_message: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None


class ResearchDecomposition(BaseModel):
    """Decomposition plan produced by the Lead Agent."""

    topic: str
    effort_level: EffortLevel
    rationale: str = Field(description="Reasoning behind effort scaling and decomposition")
    worker_count: int = Field(ge=1, le=5)
    worker_briefs: list[WorkerBrief] = Field(min_length=1)


class SourceReference(BaseModel):
    """Immutable evidence chunk in the authoritative research pool."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(description="Unique identifier of the source document or URL hash")
    chunk_id: str = Field(description="Unique identifier of the specific retrieved chunk")
    title: str = Field(description="Title of the source publication or web document")
    content: str = Field(description="Raw text content of the retrieved chunk (immutable)")
    score: float = Field(default=1.0, ge=0.0, le=1.0, description="Retrieval similarity score")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Preserved source metadata")


class CitationType(StrEnum):
    QUOTATION = "quotation"
    PARAPHRASE = "paraphrase"


class Citation(BaseModel):
    """Explicit citation linking a factual claim to an authoritative source chunk."""

    citation_id: str = Field(description="Unique citation identifier (e.g. 'cite_1')")
    source_id: str = Field(description="Referenced source document identifier")
    chunk_id: str = Field(description="Referenced source chunk identifier")
    citation_type: CitationType = Field(default=CitationType.QUOTATION)
    quoted_passage: str | None = Field(
        default=None,
        description="Verbatim excerpt from the source chunk. Mandatory if citation_type is QUOTATION.",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Claim(BaseModel):
    """An individual factual claim requiring grounding and citation."""

    claim_id: str = Field(description="Unique claim identifier, e.g. 'claim_1'")
    section_id: str = Field(description="Identifier of the section containing this claim")
    text: str = Field(description="Factual proposition asserted in the report")
    citations: list[Citation] = Field(default_factory=list)
    is_factual: bool = Field(default=True)


class ReportSection(BaseModel):
    """A distinct structural section of the research report with linked claims."""

    section_id: str = Field(description="Unique section identifier, e.g. 'sec_findings'")
    title: str = Field(description="Section heading, e.g. 'Key Findings'")
    content: str = Field(description="Full textual prose of the section with inline citations")
    claims: list[Claim] = Field(default_factory=list)


class Report(BaseModel):
    """Module 20 native research report model.

    Maintains full structural compatibility with Module 19's DeterministicVerifier
    and SemanticGroundingVerifier while adding Module 20 specific research metadata.
    """

    topic: str = Field(description="The user's original research topic or prompt")
    title: str = Field(description="Descriptive report title")
    executive_summary: ReportSection = Field(description="Executive summary section")
    sections: list[ReportSection] = Field(
        description="Body sections detailing findings and analysis"
    )
    conclusion: ReportSection = Field(description="Final synthesis section")
    limitations: list[str] = Field(
        default_factory=list,
        description="Unresolved questions, data gaps, and documented worker failures",
    )
    version: int = Field(default=1, ge=1)
    research_team_version: str = Field(
        default="1.0.0", description="Module 20 research team generator version"
    )
    brief_coverages: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of brief_id to coverage summary in the synthesized report",
    )
    unresolved_limitations: list[str] = Field(
        default_factory=list,
        description="Explicit gaps from failed or timed-out worker briefs",
    )
    created_by: str = Field(default="parallel_research_team_lead")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CheckpointState(BaseModel):
    """Serializable snapshot of the research team run for durable resume."""

    run_id: str
    user_id: str
    topic: str
    status: RunStatus
    effort_level: EffortLevel = EffortLevel.PARALLEL_DECOMPOSED
    decomposition: ResearchDecomposition | None = None
    worker_results: dict[str, WorkerExecutionResult] = Field(default_factory=dict)
    authoritative_sources: dict[str, dict[str, Any]] = Field(default_factory=dict)
    used_tokens: int = 0
    used_cost_usd: float = 0.0
    elapsed_seconds: float = 0.0
    final_report: dict[str, Any] | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
```

---

## 2. Lead Agent & Effort Scaling Business Rules

### 2.1 Decomposability & Effort Scaling Policy
1. **Rule 2.1.1 (Effort Classification):** The Lead Agent evaluates incoming questions for decomposition viability:
   - Classified as `DIRECT` if:
     - Asks for a single, isolated factual answer (e.g., "What is the capital of Peru?").
     - Is a definition or concept lookup without comparative dimensions.
     - Is a simple code snippet explanation or translation.
   - Classified as `PARALLEL_DECOMPOSED` if:
     - Involves comparative analysis across multiple systems, architectures, or vendors.
     - Requires surveying multiple distinct sub-disciplines or domains.
     - Spans multiple orthogonal research angles (e.g., historical background, performance benchmarks, and security implications).
2. **Rule 2.1.2 (Worker Count Determination):**
   - For `DIRECT` questions: `worker_count = 1`. Exactly one brief is generated.
   - For `PARALLEL_DECOMPOSED` questions:
     $$\text{worker\_count} = \max(2, \min(K, \text{config.max\_workers}, \text{config.max\_concurrency}))$$
     where $K$ is the number of naturally orthogonal sub-questions identified by the Lead Agent.
   - Default `max_workers = 4`, absolute upper ceiling $= 5$.
3. **Rule 2.1.3 (Boundary Non-Overlap Guarantee):** Every generated `WorkerBrief` must declare explicit `task_boundaries`:
   - Must contain at least one positive inclusion ("Focus strictly on...") and at least one negative exclusion ("Do NOT research...").
   - Overlap between worker scopes is strictly forbidden to prevent redundant API queries and duplicate spend.
4. **Rule 2.1.4 (Typed Output Format):** Every brief must attach an `OutputFormatSpec` specifying schema requirements (`WorkerResearchOutput`) and field guidelines, ensuring type safety and JSON-serializability for checkpoint storage.

---

## 3. Parallel Worker Execution & Concurrency Rules

### 3.1 Concurrency & Execution Envelope
1. **Rule 3.1.1 (Bounded Concurrency):** All worker dispatching is throttled through an `asyncio.Semaphore(config.max_concurrency)` (default: 3 concurrent active workers).
2. **Rule 3.1.2 (Per-Worker Step Ceiling):** Each worker is bounded by `config.per_worker_max_steps = 5` tool iterations. Any worker exceeding 5 steps terminates with `WorkerStatus.PARTIAL` and yields whatever evidence it accumulated.
3. **Rule 3.1.3 (Per-Worker Timeout):** Each worker is bounded by a monotonic wall-clock timeout (`config.per_worker_timeout_seconds = 45.0s`). If a worker hits its timeout, it is cancelled via `asyncio.wait_for` and classified as `WorkerStatus.TIMEOUT`.

### 3.2 Concurrency-Safe Shared Token Budget
1. **Rule 3.2.1 (Shared Budget Envelope):** All workers and the Lead Agent draw from a single shared run-level budget (`SharedRunBudget`):
   - Configurable defaults: `max_total_tokens = 60,000`, `max_total_cost_usd = 0.50`, `max_wall_clock_seconds = 180.0`.
2. **Rule 3.2.2 (Atomic Usage Recording):** When any worker completes an LLM completion or tool iteration, token usage and cost are recorded under an `asyncio.Lock`.
3. **Rule 3.2.3 (Proactive Budget Check):** Before every LLM completion or tool call, a worker must call `shared_budget.can_proceed()`. If the budget is exhausted, the worker aborts further calls immediately and transitions to `WorkerStatus.PARTIAL` or `FAILED`.

---

## 4. Failure Isolation & Partial Completion Rules

### 4.1 Strict Failure Isolation
1. **Rule 4.1.1 (Non-Termination of Siblings):** The failure, exception, or timeout of Worker $i$ must NEVER abort or cancel running sibling Worker $j$.
2. **Rule 4.1.2 (Exception Shielding):** All parallel tasks are gathered using `asyncio.gather(..., return_exceptions=True)`. Exceptions in individual workers are caught, logged via `safe_log`, and mapped to `WorkerExecutionResult(status=WorkerStatus.FAILED, error_message=str(exc))`.
3. **Rule 4.1.3 (Partial Evidence Salvage):** If a worker fails after executing 1 or more successful tool calls, any retrieved `SourceReference` records accumulated before the failure MUST be preserved and added to the authoritative source pool.

### 4.2 Incomplete Research Reporting
1. **Rule 4.2.1 (Mandatory Limitation Accounting):** If one or more workers fail or produce partial results, the final run status becomes `RunStatus.PARTIALLY_COMPLETED`.
2. **Rule 4.2.2 (Explicit Defect Disclosure):** The Lead Agent's final synthesized report MUST explicitly detail the failed sub-briefs, the reason for failure (timeout, network error, budget limit), and the resulting information gaps inside the report's `limitations` and `unresolved_limitations` fields.

---

## 5. Validation & Citation Verification Rules

### 5.1 Worker Output Validation
1. **Rule 5.1.1 (Structural Output Verification):** Before a worker's findings are accepted, they must pass structural validation:
   - Non-empty `findings_summary` (minimum 30 characters).
   - Valid list of `sources` containing non-empty `source_id`, `chunk_id`, and `content`.
2. **Rule 5.1.2 (Worker Citation Pointer Check):** Any claim in `WorkerResearchOutput` referencing a source must match a valid `source_id` in that worker's retrieved sources. Claims referencing unretrieved sources are flagged and discarded.

### 5.2 Evidence Transformation for `search_web`
1. **Rule 5.2.1 (Deterministic URL & Chunk Hashing):** Every snippet extracted from `search_web` is transformed into an immutable `SourceReference`:
   $$\text{source\_id} = \text{"src\_web\_"} + \text{SHA256}(\text{url})[:12]$$
   $$\text{chunk\_id} = \text{"chk\_"} + \text{SHA256}(\text{url} + \text{snippet\_text})[:12]$$
2. **Rule 5.2.2 (Verbatim Passage Alignment):** The raw `snippet` text is stored verbatim in `SourceReference.content`. Worker and synthesis prompts instruct the LLM to quote directly from this content, ensuring Gate 1 quotation substring matching succeeds deterministically.

### 5.3 Authoritative Source Pool Aggregation
1. **Rule 5.3.1 (Authoritative Evidence Pool):** Validated sources from all successful and partial workers are merged into an immutable run-level dictionary keyed by `(source_id, chunk_id)` or composite key `f"{source_id}#{chunk_id}"`.
2. **Rule 5.3.2 (Deduplication):** Redundant chunks with identical source and content are deduplicated to conserve synthesis context window tokens.

### 5.4 Lead Agent Synthesis & Verification Gates
1. **Rule 5.4.1 (Evidence-Bound Synthesis):** The Lead Agent synthesizes the report using Module 20's `Report` model (`topic`, `title`, `executive_summary`, `sections`, `conclusion`, `limitations`, `brief_coverages`, `unresolved_limitations`).
2. **Rule 5.4.2 (Citation Preservation):** Every factual claim in the synthesized report MUST be attached to one or more `Citation` objects referencing valid entries in the authoritative source pool.
3. **Rule 5.4.3 (Gate 1 Deterministic Verification):** The synthesized report is submitted to Module 19's `DeterministicVerifier`:
   - All citation pointers must resolve to known sources in the authoritative pool.
   - For citations with `CitationType.QUOTATION`, the `quoted_passage` must match a substring in the referenced source chunk after whitespace/quote normalization.
   - Reports failing Gate 1 with critical defects cannot be marked `COMPLETED`.
4. **Rule 5.4.4 (Gate 2 Semantic Grounding):** Module 19's `SemanticGroundingVerifier` verifies claim-passage entailment with a default pass threshold of **$\ge 0.70$** (configured via `ResearchTeamConfig.semantic_verification_threshold`).
5. **Rule 5.4.5 (Post-Synthesis Reconciliation):** Before passing to verification gates, `SynthesisEngine` deterministically reconciles all citations against the `AuthoritativeSourcePool`. Any citation pointing to an unknown chunk is pruned; if an altered quotation is detected, it is downgraded to a paraphrase; and if a factual claim has no remaining valid citations, its status is safely updated to `is_factual=False` to prevent Gate 1 / Gate 2 failures.

---

## 6. Memory Management, Storage & Checkpointing Rules

### 6.1 `genai-core` Memory Integration
1. **Rule 6.1.1 (Session Pinned Goals):** The top-level research goal and the decomposition plan are stored as `PinnedGoal`s in `ShortTermMemory` under the user's active session.
2. **Rule 6.1.2 (Long-Term User Context):** Before decomposition, the Lead Agent queries `LongTermMemory` for user domain preferences, constraints, or prior guidelines, including them in the system prompt.
3. **Rule 6.1.3 (Episodic Research Archival):** Upon successful completion, the completed research run is recorded in `EpisodicMemory` with its canonical question hash to enable idempotent retrieval.

### 6.2 Dual Retrieval Storage Support
1. **Rule 6.2.1 (Storage Backend Selection):** `RetrievalService` supports dual modes governed by `config.retrieval_mode`:
   - `retrieval_mode == "postgres"`: Uses `PGVectorStoreAdapter` with connection string specified by `config.postgres_connection_string`.
   - `retrieval_mode == "in_memory"`: Uses `InMemoryVectorStore` for fast, zero-dependency offline testing and local execution.

### 6.3 Real-Time Checkpointing & Resume Protocol
1. **Rule 6.3.1 (Stateful Persistence):** A `CheckpointState` is saved to `CheckpointStore` immediately at three critical milestones:
   - Milestone A: Post-decomposition (plan and briefs created).
   - Milestone B: As each individual worker completes its task.
   - Milestone C: Post-synthesis and verification.
2. **Rule 6.3.2 (Idempotent Resumption):** When `resume(run_id)` is invoked:
   - The checkpoint is retrieved from `CheckpointStore`.
   - Completed workers (`WorkerStatus.COMPLETED`) are NEVER re-executed; their outputs and sources are preserved.
   - Remaining tokens and budget are recalculated: $\text{remaining\_tokens} = \text{max\_tokens} - \text{used\_tokens}$.
   - Only workers in `PENDING`, `RUNNING` (interrupted mid-flight), or retryable `FAILED` states are dispatched.
   - If all workers are completed, the system proceeds directly to synthesis and verification.

---

## 7. Tracing, Observability & Cost Tracking Rules

### 7.1 Telemetry Isolation & Cost Accounting
1. **Rule 7.1.1 (Context Propagation):** Every worker execution runs within a bound `observability_context` bearing the global `run_id` and individual `worker_id` / `brief_id`.
2. **Rule 7.1.2 (Safe Logging):** All worker lifecycle events, tool invocations, and errors must be emitted via `safe_log` using standard `ObservabilityEvent` constants.
3. **Rule 7.1.3 (Native Pricing Calculator):** Token consumption (prompt and completion tokens) for each model call is converted to USD using Module 20's `PricingCalculator` (`src/parallel_research_team/telemetry/pricing.py`).
4. **Rule 7.1.4 (Consolidated Run Summary):** Every completed or failed run outputs an `ExecutionMetrics` summary reporting:
   - Total tokens consumed (Lead Agent vs Workers vs Synthesis).
   - Total USD cost.
   - Wall-clock elapsed duration.
   - Worker breakdown: total, completed, partial, failed, and timed-out.
5. **Rule 7.1.5 (Graceful Budget Exhaustion Interception):** When `BudgetExceededError` occurs, the orchestrator MUST NOT crash the process. It must transition the run to `RunStatus.PARTIALLY_COMPLETED`, preserve all gathered evidence and partial deliverables, synthesize a partial report if evidence exists, and record a user-friendly explanatory disclosure in `report.limitations` and `report.unresolved_limitations`.


---

## 8. Comparative Evaluation & Benchmark Protocol

### 8.1 Benchmark Dataset Governance
1. **Rule 8.1.1 (Archetype Distribution):** The benchmark suite must contain at least 15 curated questions spanning 4 archetypes:
   - Simple Factual (4 questions): Atomic single-fact queries. Single agent must outperform or match multi-agent.
   - Linear Synthesis (3 questions): Stepwise reasoning queries. Multi-agent fan-out should not provide quality advantage.
   - Multi-Faceted Comparison (4 questions): Multiple orthogonal dimensions. Multi-agent should win on coverage and trade-off depth.
   - Architectural Survey (4 questions): Full-stack systems architecture. Multi-agent should win on comprehensiveness and structural clarity.
2. **Rule 8.1.2 (Fair Comparison Tool Equivalence):** The Single Agent baseline (Module 17 `BoundedAgent`) must be supplied with the exact same search tools as Module 20 (`search_web` via Tavily and `search_knowledge_base` via the handbook vector store) to ensure pure architectural comparison.

### 8.2 Rate Limiting & Cooldown Protection
1. **Rule 8.2.1 (Hard RPM Ceiling):** All LLM requests during evaluation must be guarded by `AsyncRateLimiter` with a maximum ceiling of $\le 13.3$ RPM (`min_interval_seconds = 4.5s`) to prevent HTTP 429 rate limit errors on Gemini free tiers.
2. **Rule 8.2.2 (Mandatory Inter-Question Break):** A mandatory 30-second cooldown pause (`await asyncio.sleep(30.0)`) must be enforced between benchmark questions.
3. **Rule 8.2.3 (Atomic Progress Checkpointing):** Each question result must be appended and persisted to `eval_results/eval_run_<run_id>.json` immediately upon completion.
