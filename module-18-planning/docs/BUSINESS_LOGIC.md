# Module 18: Plan-and-Execute Research Assistant — Confirmed Business Logic

## 1. Domain Models, States & Schemas

### 1.1 Execution States & Enums

#### `StepStatus`
Governs the lifecycle of an individual step in the plan:
- `PENDING`: Initial state upon plan generation. Dependencies not yet satisfied.
- `READY`: All predecessor step IDs in `depends_on` have transitioned to `COMPLETED`. Step is eligible for dispatch.
- `RUNNING`: Step is currently executing asynchronously in `StepExecutor`.
- `COMPLETED`: Step completed execution and passed both programmatic and model-based verification.
- `FAILED`: Step execution failed, or verification failed after exhausting step-level retries.
- `BLOCKED`: Step cannot run because one or more direct or indirect dependencies failed and could not be resolved by replanning.
- `SKIPPED`: Step was explicitly bypassed during dynamic replanning because an alternative execution path was selected.

#### `RunStatus`
Governs the end-to-end assistant execution lifecycle:
- `INITIALIZING`: Ingesting goal and recalling user preferences and episodic memory.
- `PLANNING`: Generating structured DAG plan and performing validation/repair loops.
- `REVIEW_PENDING`: Awaiting human review decision before execution.
- `EXECUTING`: Dispatched to `DAGScheduler` for bounded, concurrent execution.
- `REPLANNING`: Pausing dispatch to revise the plan following step failure while preserving completed work.
- `COMPLETED`: All required plan steps reached `COMPLETED` and the final verified deliverable was synthesized.
- `FAILED`: Run halted due to exhausted replan limit, unrecoverable DAG deadlock, or global budget breach.
- `CANCELLED`: Execution explicitly terminated by operator rejection during human review.

#### `ReviewAction`
- `APPROVE`: Human reviewer accepts the generated plan without modifications.
- `MODIFY`: Human reviewer provides an edited plan. Triggers DAG re-validation and increments plan version.
- `REJECT`: Human reviewer rejects the plan. Aborts run or triggers complete re-generation.

---

### 1.2 Core Data Schemas

```python
class PlanStep(BaseModel):
    id: str = Field(description="Unique step identifier, e.g. 'step_1'")
    description: str = Field(description="Actionable objective for the agent")
    tool_name_hint: str | None = Field(default=None, description="Suggested tool category/domain")
    depends_on: list[str] | None = Field(default=None, description="List of predecessor step IDs")
    success_criteria: str = Field(description="Explicit, verifiable success condition")
    status: StepStatus = Field(default=StepStatus.PENDING)
    result_summary: str | None = Field(default=None, description="Output summary after completion")
    attempts: int = Field(default=0, description="Number of execution attempts")
    error_message: str | None = Field(default=None, description="Error detail if failed")


class Plan(BaseModel):
    goal: str = Field(description="Top-level user objective")
    steps: list[PlanStep] = Field(description="List of planned steps forming a DAG")
    version: int = Field(default=1, description="Monotonically increasing plan version")
    parent_version: int | None = Field(default=None, description="Version ID of predecessor plan")
    created_by: str = Field(default="planner_llm")  # "planner_llm", "human_reviewer", "replanner"
    assumptions: list[str] = Field(
        default_factory=list, description="Explicit planning assumptions"
    )
    change_summary: str | None = Field(
        default=None, description="Summary of edits from parent version"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AssistantRunResult(BaseModel):
    """Consolidated outcome returned by PlanAndExecuteAssistant."""

    run_id: str = Field(description="Unique execution run identifier")
    status: RunStatus = Field(description="Final execution lifecycle status")
    final_deliverable: str | None = Field(
        default=None, description="Synthesized final deliverable text"
    )
    plan: Plan = Field(description="Final Plan DAG associated with the run")
    run_record: RunRecord = Field(description="Full audit record containing all step traces")
    total_tokens: int = Field(default=0, ge=0, description="Cumulative tokens consumed")
    total_cost_usd: float = Field(
        default=0.0, ge=0.0, description="Cumulative cost incurred in USD"
    )
    duration_ms: float = Field(
        default=0.0, ge=0.0, description="Total wall-clock execution time in ms"
    )
    error_message: str | None = Field(default=None, description="Error detail if execution failed")


class ReflectionResult(BaseModel):
    """Structured result returned by a PlanReflector evaluation."""

    should_replan: bool = Field(description="Whether reflection advises replanning")
    critique: str = Field(description="Detailed reflection analysis and critique")
    suggested_improvements: list[str] = Field(
        default_factory=list, description="Specific recommendations for plan revision"
    )
    score: float = Field(default=1.0, ge=0.0, le=1.0, description="Plan execution quality score")
```

---

## 2. Structured Planning & DAG Validation Rules

### 2.1 Structural Integrity Constraints
Every `Plan` generated by LLM or submitted by human review must strictly satisfy five validation invariants enforced by `PlanValidator`:

1. **Non-Empty Plan:** The plan must contain at least one step (`len(plan.steps) >= 1`). Plans with zero steps raise `EmptyPlanError`.
2. **Unique Step Identifiers:** Every step must have a unique `id`. Duplicate step IDs raise `DuplicateStepIdError`.
3. **No Self-Dependencies:** A step cannot list its own `id` in `depends_on`. Direct self-reference raises `SelfDependencyError`.
4. **Referential Integrity:** Every ID declared in `depends_on` must exist in the plan's `steps`. Referencing a non-existent step ID raises `MissingDependencyError`.
5. **Acyclicity (DAG Invariant):** The dependency graph must contain zero directed cycles. Acyclicity is verified using a 3-state Depth-First Search (`0=UNVISITED`, `1=VISITING`, `2=VISITED`). Encountering a node in `VISITING` state indicates a directed cycle and raises `CircularDependencyError` displaying the exact cycle path (e.g. `step_1 -> step_2 -> step_1`).
6. **Root Reachability:** At least one step must have `depends_on=None` or `depends_on=[]` to ensure execution can begin without deadlock.

### 2.2 Automated Plan Generation & Repair Loop
- **Model Invocations:** The planner invokes `genai_core.LLMService.complete_structure(prompt=..., response_schema=Plan)` targeting `gemini-3.5-flash-lite`.
- **Dynamic Capability Discovery:** The planner queries `StepExecutor.get_available_capabilities()` upon initialization to obtain `{name: description}` for all registered executor tools. This capability catalogue is dynamically injected into the planning prompt so the LLM assigns valid `tool_name_hint` values without hardcoding tool registries.
- **Validation Retry Policy:** If `complete_structure` returns an invalid DAG that raises `PlanValidationError`:
  1. The validator captures the exact error type, offending step ID, and cycle/missing dependency trace.
  2. The `PlanRepairer` formats a corrective prompt including the full validation error and available capabilities.
  3. The repair loop runs up to `max_planning_retries` (default: 3). If all retries fail, planning halts with `PlanGenerationError`.

### 2.3 Memory Integration & Configurable Episodic Recall
- **Long-Term Memory Preferences:** The planner automatically recalls user preferences and domain constraints from `genai_core.LongTermMemory` via `search_longterm(user_id=..., query=goal)`.
- **Config-Gated Episodic Past Plans:**
  - Governed by `AssistantConfig.enable_episodic_similarity_search: bool = False`.
  - When `False` (default): Episodic search is cleanly bypassed. The planner operates in zero-shot mode, avoiding unnecessary latency and token overhead.
  - When `True`: The memory manager queries semantic episodic search if available on `MemoryService`, falling back to recent `ActionEpisode`s filtered by `action_type="plan_execution"`.
  - When `genai-core` introduces semantic episodic vector search in a future release, enabling it in Module 18 requires only setting this config toggle to `True`.
- **Episodic Persistence:** Completed plan executions are recorded via `record_plan_episode()` into `genai_core.EpisodicMemory` with the goal, step summaries, status, and execution metadata for auditability and future few-shot retrieval.

---

## 3. Human Plan Review & Versioning Business Rules

### 3.1 Review Gating Policy
- Autonomous execution of a newly generated plan is strictly prohibited without human operator sign-off.
- The assistant presents the initial plan (`version=1`) with its steps, descriptions, dependencies, tool hints, and success criteria to the operator.
- **Automated / CI Bypass (`auto_approve_review`):** For automated test suites, CI/CD pipelines, and evaluation benchmarks, callers may pass `auto_approve_review=True` in `AssistantConfig` to simulate immediate operator approval without blocking on stdin.

### 3.2 Review Decision Handling & Immutability
- **`ReviewAction.APPROVE`:** The plan proceeds immediately to `DAGScheduler` with `version=1`.
- **`ReviewAction.MODIFY`:**
  - The reviewer provides a modified `Plan`.
  - The modified plan is re-validated via `PlanValidator`. If invalid, the operator is prompted to fix the validation errors.
  - Upon passing validation, the modified plan is stamped as a new version:
    - `version = parent_version + 1` (e.g., `version=2`).
    - `parent_version = 1`.
    - `created_by = "human_reviewer"`.
    - `change_summary = reviewer_notes`.
  - Previous plan versions remain immutable and are permanently archived in the run record.
- **`ReviewAction.REJECT`:** Execution halts cleanly, transitioning `RunStatus` to `CANCELLED`. The rejection reason is recorded in the run record.

---

## 4. DAG Scheduling & Concurrency Control Rules

### 4.1 Dependency Resolution & State Transitions
- The `DAGScheduler` manages execution as an asynchronous event-driven state machine.
- A step transitions from `PENDING` to `READY` if and only if:
  $$\forall d \in \text{step.depends\_on}, \quad \text{status}(d) == \text{COMPLETED}$$
- Root steps (`depends_on == []` or `None`) are immediately marked `READY`.

### 4.2 Bounded Concurrency
- Ready steps are dispatched concurrently as individual `asyncio.Task` instances.
- Concurrency is strictly bounded by `asyncio.Semaphore(max_concurrency)` (default: 3).
- No more than `max_concurrency` agent execution steps may be running concurrently. This protects against API rate limit spikes (e.g. Gemini 15 RPM limits), token burst exhaustion, and excessive concurrent database connections.

### 4.3 Predecessor Context Propagation
- When dispatching a `READY` step, the scheduler extracts the `result_summary` and verified outputs from all steps listed in its `depends_on`.
- Predecessor results are formatted into the step prompt:
  ```
  Predecessor Context:
  - [{dep_id}]: {result_summary}
  ```
- Steps without dependencies receive clean initial context without extraneous predecessor history.

---

## 5. Bounded Step Execution & Dual Verification Rules

### 5.1 Step Execution via Module 17 (`BoundedAgent`)
- Individual plan steps are executed through `research_and_support_agent.BoundedAgent`.
- Each step is allocated a focused step-level budget:
  - `max_steps`: default 5 iterations per step.
  - `max_tokens`: default 4,000 tokens per step.
  - `max_cost_usd`: default $0.02 USD per step.
  - `timeout_seconds`: default 45 seconds monotonic.
- If an irreversible tool (e.g. `execute_refund`, `delete_customer_data`) is triggered during step execution, Module 17's native `PostgresApprovalStore` handles the human authorization pause/resume transparently.

### 5.2 Dual Verification Gate
Before any step transitions to `COMPLETED`, it must pass two independent verification gates:

```
Step Execution Finished
          |
          v
+-------------------------------+
|  1. Programmatic Verifier     | ---> Fails ---> Mark Step Failed / Retry
+-------------------------------+
          | Passes
          v
+-------------------------------+
|  2. Model-Based Judge Verifier| ---> Fails ---> Mark Step Failed / Retry
+-------------------------------+
          | Passes
          v
   Mark Step COMPLETED
```

#### Gate 1: Programmatic Verifier
Fast, deterministic assertion checking:
1. `AgentRunResult.status == ToolLoopStatus.COMPLETED`. (Runs that halted with `BUDGET_EXCEEDED` or `LOOP_DETECTED` fail immediately).
2. `final_text` is non-empty and contains at least 20 characters.
3. If the step produces structured data or file artifacts, the programmatic verifier runs schema validation or file existence checks.

#### Gate 2: Model-Based Judge Verifier
Semantic evaluation using `genai_core.LLMService.complete_structure`:
- **Judge Schema:**
  ```python
  class StepJudgeVerdict(BaseModel):
      is_satisfied: bool = Field(description="True if actual output satisfies the success criteria")
      score: float = Field(ge=0.0, le=1.0, description="Verification confidence score")
      reasoning: str = Field(description="Objective justification for the score")
      missing_criteria: list[str] = Field(default_factory=list)
  ```
- **Threshold Rule:** The step passes model verification if:
  $$\text{verdict.is\_satisfied} == \text{True} \quad \land \quad \text{verdict.score} \ge 0.85$$

### 5.3 Step-Level Retries
- If verification fails (either programmatic or model judge), the step is not immediately escalated to replanning.
- The step is retried locally up to `max_step_retries` (default: 2).
- The retry prompt includes the verifier's specific feedback:
  ```
  Previous attempt failed verification:
  Reasoning: {verdict.reasoning}
  Missing Criteria: {verdict.missing_criteria}
  Please address the missing criteria and complete the step.
  ```
- If the step still fails after `max_step_retries`, its status transitions to `FAILED`, and the orchestrator triggers dynamic replanning.

---

## 6. Concurrency-Safe Dynamic Replanning Rules

When a step fails, the assistant executes a **concurrency-safe dynamic replanning protocol**:

```mermaid
flowchart TD
    A[Step X Fails Verification] --> B[Pause Dispatching New Steps]
    B --> C[Await Currently Running Sibling Steps]
    C --> D[Collect Sibling Results & Update Cache]
    D --> E[Isolate Completed Steps & Preserve Them]
    E --> F[Generate Revised Plan via ReplanningEngine]
    F --> G[Validate Revised Plan as DAG]
    G --> H[Increment Plan Version N+1 with Parent N]
    H --> I[Resume DAG Dispatch for Remaining Steps]
```

### 6.1 Concurrency-Safe Sibling Handling Policy
When parallel steps are executing concurrently (e.g. Step A and Step B running simultaneously) and Step A fails:
1. **Immediate Ingestion Pause:** The scheduler immediately stops dispatching any new `READY` steps.
2. **Sibling Settling:** The scheduler does *not* abruptly kill or discard currently running sibling Step B. Instead, Step B is allowed to finish its active execution and verification.
3. **Cache Consolidation:** If Step B succeeds, its `result_summary` and status are permanently recorded as `COMPLETED`. This prevents throwing away expensive LLM and tool work already paid for.
4. **Clean Settle:** Once all actively running sibling tasks settle, replanning begins with the complete, settled state.

### 6.2 Work Preservation Invariant
- **Never Re-execute Completed Work:** Steps that have achieved `COMPLETED` status prior to replanning are immutable.
- In the replanned DAG, completed steps retain their original step IDs, cached outputs, and `COMPLETED` statuses.
- The replanner is strictly forbidden from resetting a completed step to `PENDING` unless an explicit conflict is identified.

### 6.3 Subgraph Replanning Rules
- The `ReplanningEngine` feeds the following context into the replanning prompt:
  1. Top-level goal.
  2. List of `COMPLETED` steps (marked immutable).
  3. The `FAILED` step, its error trace, and the verifier's critique.
  4. Remaining unfulfilled objectives.
- The replanner generates replacement, remedial, or alternative steps to fulfill the overall goal without repeating completed work.
- The resulting plan is verified via `PlanValidator`.
- The new plan is stamped:
  - `version = parent_version + 1`.
  - `parent_version = current_version`.
  - `created_by = "replanner"`.
  - `change_summary = f"Replanned after failure of {failed_step_id}: {reason}"`.

### 6.4 Replanning Ceilings
- Replanning is permitted up to `max_replans` (default: 3) per goal execution.
- If a run triggers more than `max_replans`, the run terminates with `RunStatus.FAILED` and error reason `REPLAN_LIMIT_EXCEEDED` to prevent infinite replanning loops and runaway token expenditure.

---

## 7. Run-Wide Budget Rules

Module 18 introduces a global, run-wide budget envelope (`RunBudget`) that spans the entire assistant lifecycle, including initial planning, human review latency, parallel step execution, verification evaluations, step retries, and replanning iterations.

### 7.1 Budget Parameters & Defaults

| Budget Parameter | Default Ceiling | Scope & Enforcement |
| :--- | :--- | :--- |
| `max_total_tokens` | 40,000 tokens | Cumulative prompt and completion tokens across all LLM calls. |
| `max_total_cost_usd` | $0.25 USD | Cumulative dollar cost calculated via `PricingCalculator`. |
| `total_timeout_seconds` | 180 seconds | Monotonic wall-clock deadline (`time.monotonic() + total_timeout_seconds`). |
| `max_concurrency` | 3 parallel steps | Maximum concurrent active step tasks in `DAGScheduler`. |
| `max_planning_retries`| 3 attempts | Maximum structural repair attempts during initial planning. |
| `max_step_retries` | 2 attempts | Maximum retry attempts per individual step before replanning. |
| `max_replans` | 3 replans | Maximum replanning cycles per overall run. |

### 7.2 Budget Exhaustion Behavior
- Before dispatching any step or invoking any LLM call (planning, verification, replanning, reflection, deliverable synthesis), the orchestrator checks `RunBudget.check_budget()`.
- If cumulative tokens exceed `max_total_tokens`, cumulative cost exceeds `max_total_cost_usd`, or wall-clock time exceeds `total_timeout_seconds`, `RunBudget.check_budget()` raises `BudgetExceededError`.
- The assistant catches `BudgetExceededError`, logs an observability error event, stamps `run_record.status = RunStatus.FAILED` with error message `"Budget exceeded: ..."`, flushes the run record atomically via `FileRunRecordStore`, and returns `AssistantRunResult`.

---

## 8. Memory Subsystem Integration Rules

The assistant integrates with `packages/genai-core`'s memory architecture across two distinct memory tiers:

### 8.1 Long-Term Memory (User Preferences & Operational Constraints)
- **Ingestion & Retrieval:** Uses `genai_core.MemoryService.search_longterm(user_id=user_id, query=goal, limit=5)`.
- **Ranking:** Utilizes `genai_core`'s hybrid cosine similarity and exponential recency decay ($t_{1/2} = 30$ days).
- **Prompt Injection:** Recalled user facts and preferences (e.g., preferred technologies, output formatting, risk constraints) are injected into the system prompt of the `PlanGenerator` to ensure plans conform to user expectations.

### 8.2 Episodic Memory (Action Logging & Semantic Similarity Search)
- **Current Operational Baseline (Module 18):**
  - **Action Logging:** Upon goal completion, the assistant records an `ActionEpisode` via `genai_core.EpisodicMemory.record()` with:
    - `action_type = "plan_execution"`
    - `parameters = {"goal": goal, "plan_version": plan.version, "step_count": len(plan.steps)}`
    - `result_summary = final_deliverable_summary`
    - `status = ActionStatus.SUCCESS`
    - `metadata = {"total_cost_usd": run_record.total_cost_usd, "total_tokens": run_record.total_tokens}`
  - **Audit & Recency Recall:** Module 18 retrieves recent episodes for the user via `get_episodes(user_id, action_type="plan_execution")`.
- **Planned Capability in `genai-core` (Semantic Similarity Search):**
  - Semantic vector similarity search over episodic memory (`search_episodes(user_id, query, limit=5)`) is planned for future addition directly to `packages/genai-core`.
  - **Module 18 Consumer Interface:** Module 18 defines the consumer contract in `PlanningMemoryManager.get_similar_past_plans(user_id, goal)`. Until the feature is implemented in `genai-core`, this method retrieves recent plan episodes for few-shot prompting. When `search_episodes` is added to `genai-core`, Module 18 will delegate directly to it with zero breaking interface changes.

---

## 9. Run Records & Telemetry Business Rules

### 9.1 Data Completeness Invariant
Every plan run produces an auditable `RunRecord` containing:
- Unique `run_id` (UUIDv4) and `user_id`.
- Initial goal and final synthesized deliverable.
- Chronological list of all `Plan` versions (`plan_versions`).
- Detailed execution history for every step (`step_executions: dict[str, list[StepExecutionRecord]]`):
  - Step ID, plan version, dependencies, assigned tool hints.
  - Execution attempts, duration ms, input/output tokens, and cost.
  - Full verification verdict, score, reasoning, and missing criteria.
- Overall execution status (`RunStatus`), total tokens, total cost ($ USD), and wall-clock duration ms.

### 9.2 Persistence & Storage
- Run records are serialized to JSON and saved locally to `runs/run_{run_id}.json` via `FileRunRecordStore`.
- Conforms to the `RunRecordStore` protocol, allowing future seamless migration to PostgreSQL without modifying orchestration code.

---

## 10. Fair Comparative Evaluation Benchmark Rules

To scientifically validate Module 18's architecture, a rigorous 10-goal benchmark compares the **Planned Approach** (Module 18) against the **Reactive Baseline** (Module 17 `BoundedAgent`).

### 10.1 Ground Rules for Fair Comparison
To prevent evaluation bias, both systems are benchmarked under strictly equivalent environmental constraints:
1. **Identical Goals:** Both systems evaluate the exact same 10 curated goals.
2. **Identical Model Configuration:** Both systems execute using Google Gemini 3.5 Flash Lite (`gemini-3.5-flash-lite`) with identical temperature ($0.2$) and rate limiting.
3. **Equalized Resource Caps:** Both systems are constrained by identical total token ($40,000$ tokens) and cost ($0.25$ USD) budgets per goal.
4. **Identical Tool & Database State:** Both systems share access to the same 16 tools, identical PostgreSQL seed tables (`support_customers`, `support_tickets`), and the same knowledge base chunks.
5. **Uniform Success Criteria:** A single, neutral model-based judge (`genai_core.LLMService.complete_structure`) evaluates the final output of both systems against identical ground-truth reference requirements.

### 10.2 The 10 Curated Benchmark Goals

| Goal ID | Category | Goal Statement | Key Architectural Test |
| :--- | :--- | :--- | :--- |
| **G-01** | Research | Compare HNSW and IVFFlat vector indexing algorithms in terms of latency, build time, and recall. | Parallel independent research branches joining at synthesis. |
| **G-02** | Operations | Investigate customer account C-101, inspect recent billing tickets, verify refund policy, and draft customer response. | Strict sequential order dependencies across DB and KB. |
| **G-03** | Operations | Resolve high-priority ticket T-102 and update customer account tier if eligible. | Gated operational sequence requiring inspection before status write. |
| **G-04** | Research | Perform security audit comparing internal knowledge-base guidelines with public OWASP Top 10 web vulnerabilities. | Multi-query KB retrieval combined with external web search. |
| **G-05** | Mixed | Diagnose customer churn risk for customer C-104 using ticket history, stored memory preferences, and account tier. | Multi-tier reasoning combining database and long-term memory. |
| **G-06** | Resilience | Retrieve documentation for an ambiguous query requiring dynamic replanning after initial verification failure. | Tests verifier detecting insufficient output and triggering replan. |
| **G-07** | Operations | Batch status and resolution audit across tickets T-101, T-103, and T-105 with a single consolidated executive brief. | 3 independent parallel fetch steps merging into 1 synthesis step. |
| **G-08** | Research | Analyze pricing models and SLA terms for three cloud vector databases from web sources with a markdown comparison matrix. | Multi-source web search with structured table deliverable. |
| **G-09** | Operations | Prepare refund execution: verify customer eligibility, calculate prorated refund amount, and record audit log. | Verification of eligibility prerequisites prior to irreversible action. |
| **G-10** | Mixed | Root-cause analysis for system incident: search error log chunks, query incident ticket, and draft postmortem. | Multi-hop reasoning where subsequent step depends on initial discovery. |

### 10.3 Evaluation Metrics
The benchmark CLI (`scripts/run_eval.py`) computes and reports:
- **Completion Rate (%):** Percentage of goals meeting verified ground-truth success criteria.
- **Mean Steps / Turns:** Average LLM/tool turns required per goal.
- **Mean Cost ($ USD):** Average dollar cost per goal using actual token usage.
- **Mean Latency (seconds):** Total wall-clock time per goal (highlighting parallel DAG speedups).
- **Replanning Recovery Rate (%):** Percentage of step failures successfully recovered by dynamic replanning.

---

## 11. Extensibility Contracts for Modules 19 & 20

Module 18 defines clean Protocol boundaries to ensure future modules integrate without modifying core planning logic:

### 11.1 Module 19: Plan Reflection Protocol
```python
@runtime_checkable
class PlanReflector(Protocol):
    """Module 19 Hook: Evaluates plan execution and generates reflective critique."""

    async def reflect(
        self,
        plan: Plan,
        step_executions: dict[str, list[StepExecutionRecord]],
        user_id: str,
    ) -> ReflectionResult: ...
```
- In Module 18, `reflector` defaults to `None`. When provided in Module 19, `PlanAndExecuteAssistant` passes the final plan and step execution history to `reflector.reflect()` prior to deliverable synthesis.

### 11.2 Module 20: Specialist Agent Delegation Protocol
```python
class StepExecutor(Protocol):
    """Module 20 Hook: Routes step execution to specialist agents based on domain or tool hints."""

    async def execute_step(
        self,
        step: PlanStep,
        predecessor_context: dict[str, str],
        user_id: str,
    ) -> StepExecutionResult: ...
```
- In Module 18, the default implementation is `Module17StepExecutor`, routing all steps to `BoundedAgent`.
- In Module 20, a `SpecialistAgentRouter` can implement `StepExecutor` to dispatch code steps, research steps, and database steps to specialized agents based on `step.tool_name_hint`.

---

## 12. Live Provider & Database Integration Architecture

### 12.1 Live LLM Provider Configuration
- **Provider:** Google Gemini API via `genai_core.GoogleProvider`.
- **Active Model:** `gemini-3.5-flash-lite` (designated default due to deprecation of `gemini-2.0-flash` for new users).
- **Fallback Models:** `gemini-3.8-flash`.

### 12.2 Containerized Database Architecture
- **Engine:** PostgreSQL 17 with `pgvector` extension via `module-18-planning/docker-compose.yml`.
- **Connection URL:** `postgresql://postgres:postgres@127.0.0.1:5432/ai_search`.
- **Managed Schemas:**
  - `support_customers`: Customer CRM profile (ID, name, email, tier, balance, risk flag).
  - `support_tickets`: Support tickets (ID, customer ID, subject, status, priority, description).
  - `document_chunks`: pgvector semantic embeddings for knowledge base retrieval (384-dimensional `all-MiniLM-L6-v2`).
  - `memory_lt_records`: Long-term preference records for personalization.
  - `memory_ep_episodes`: Episodic execution memory for past plan recall and audit logging.

### 12.3 Test Suite Partitioning
- **Unit Tests (`tests/unit/`):** Purely isolated, 0 external network calls, fast execution (`pytest tests/unit/`), executed during pre-push quality gates.
- **Mock Integration Tests (`tests/integration/test_eval_benchmark.py`):** Deterministic mock execution for fast 10-goal benchmarking.
- **Live Integration Tests (`tests/integration/test_live_*.py`):** Gated by `INTEGRATION_TEST=1`, executed against real Gemini API and PostgreSQL containers.
