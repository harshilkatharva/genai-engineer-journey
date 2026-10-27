# Module 18: Plan-and-Execute Research Assistant — Architectural Decisions

## 2026-10-26: Initial Architecture, DAG Planning, and Evaluation Design

### Decision 1: Pure Python `asyncio` DAG Scheduler with Semaphore Concurrency (vs. LangGraph)
- **Context:** The assistant requires an asynchronous DAG scheduler that can evaluate step dependency readiness, execute independent ready steps in parallel, enforce a strict concurrency cap, and handle dynamic replanning. `pyproject.toml` previously listed `langgraph>=1.2.15`.
- **Decision:** Implement a pure Python asynchronous scheduler (`DAGScheduler`) using native `asyncio`, `asyncio.Semaphore(max_concurrency)`, and topological graph resolution, without relying on LangGraph or heavy graph frameworks.
- **Alternatives Considered:**
  1. *LangGraph:* While LangGraph supports graph-based flows, it introduces framework-specific abstractions, complex state reducer boilerplate, non-trivial human pause/resume wiring, and opaque debugging.
  2. *Celery / Airflow / Temporal:* Heavyweight orchestration engines requiring distributed brokers, out of scope for a bounded local Python agent package.
- **Impact:** Aligns with the repository's foundational principle ("Framework-aware, not framework-dependent"). Yields zero framework overhead, complete visibility into task scheduling, clean coroutine cancellation, trivial mocking in unit tests, and seamless extensibility for Modules 19 and 20.

---

### Decision 2: Dual Verification Architecture (Deterministic Programmatic + Model-Based Judge)
- **Context:** In multi-step planning, an agent's self-reported success cannot be trusted blindly. Prematurely marking a flawed step as completed causes cascading failures in downstream dependent steps.
- **Decision:** Mandate a two-stage verification gate for every completed step:
  1. *Programmatic Verifier:* Validates hard invariants (status is `COMPLETED`, deliverable text is non-empty with minimum character count, deterministic schema/regex validation).
  2. *Model-Based Judge Verifier:* Uses `genai_core.LLMService.complete_structure` with a structured `StepJudgeVerdict` schema to objectively evaluate whether the actual deliverable satisfies the step's explicit `success_criteria` (threshold >= 0.85).
- **Alternatives Considered:**
  1. *LLM-only verification:* Expensive and susceptible to hallucinated success if the agent output is empty or errored.
  2. *Programmatic-only verification:* Incapable of verifying semantic correctness, nuance, or qualitative criteria in research tasks.
- **Impact:** Eliminates false positives, ensures high output fidelity across DAG steps, and provides actionable failure critique to guide step-level retries and dynamic replanning.

---

### Decision 3: Concurrency-Safe Dynamic Replanning with Sibling Settle & Completed Work Preservation
- **Context:** When independent steps run concurrently and one step fails, naive orchestrators either abruptly kill valid running sibling tasks or re-execute previously completed work during replanning, wasting tokens and wall-clock time.
- **Decision:** Implement a concurrency-safe replanning protocol:
  1. When Step A fails, immediately halt dispatch of *new* steps.
  2. Allow currently running sibling tasks (e.g. Step B) to complete active execution and verification.
  3. Consolidate and preserve all `COMPLETED` steps in the execution cache; never re-execute them.
  4. Invoke `ReplanningEngine` with completed steps marked immutable, focusing the replanner exclusively on generating replacement or remedial steps for the failed branch.
  5. Increment the plan version, link `parent_version`, and resume scheduling for unfulfilled steps.
- **Alternatives Considered:**
  1. *Abrupt sibling cancellation:* Kills tasks mid-flight, discarding expensive LLM generation and tool execution.
  2. *Full-plan re-generation:* Discards completed work and starts from scratch, increasing latency and cost.
- **Impact:** Minimizes wasted execution, prevents duplicate tool side-effects, guarantees thread/coroutine safety during plan mutation, and preserves valuable partial progress.

---

### Decision 4: Global Run-Wide Budget Envelope Spanning All Lifecycle Phases
- **Context:** Module 17 enforced step-level budgets on `BoundedAgent`. However, in a multi-step planning assistant with parallel execution, LLM verification judges, step retries, and dynamic replanning loops, step-level budgets alone cannot prevent cumulative cost and token overruns.
- **Decision:** Introduce a top-level `RunBudget` tracking aggregate token consumption, total USD cost, and a monotonic wall-clock deadline across the entire assistant lifecycle (planning, review, step execution, verification, and replanning). All phase components deduct from this shared envelope and abort execution immediately if exhausted.
- **Alternatives Considered:** Isolated per-phase budgets (e.g. separate budgets for planning, execution, and verification). Rejected because it allows cumulative cost to balloon if multiple retries and replans occur across phases.
- **Impact:** Guarantees strict financial and operational safety bounds ($0.25 USD, 40,000 tokens, 180s wall-clock default), preventing runaway loops or denial-of-wallet scenarios.

---

### Decision 5: Immutable Plan Versioning and Append-Only History in Run Records
- **Context:** The plan evolves through multiple stages: initial generation, human review modifications, and dynamic replanning. Debugging, postmortems, and evaluation require complete auditability of what changed, why, and by whom.
- **Decision:** Treat plans as immutable data structures. Every modification creates a new `Plan` instance with `version = parent_version + 1`, a `parent_version` pointer, a `created_by` author tag (`"planner_llm"`, `"human_reviewer"`, `"replanner"`), and a descriptive `change_summary`. All versions are preserved in the run record (`RunRecord.plan_versions`).
- **Alternatives Considered:** In-place mutable plan objects. Rejected because in-place mutation obscures the origin of plan changes, breaks race condition safety during concurrent step execution, and prevents post-run lineage analysis.
- **Impact:** Provides 100% auditability, transparent diffing between human and agent plans, and safe concurrency semantics.

---

### Decision 6: Human Plan Review Gateway with Dual Interactive/Automated Modes
- **Context:** Production planning assistants require human-in-the-loop validation before autonomous step dispatch. However, mandatory interactive stdin prompts break automated test suites, CI/CD runners, and benchmark scripts.
- **Decision:** Implement `HumanPlanReviewGateway` with two operational modes:
  1. *Interactive Mode (Default):* Prompts the operator via CLI to approve, modify, or reject the proposed plan.
  2. *Automated Mode (`auto_approve_review=True`):* Automatically approves generated plans, allowing programmatic test execution, regression suites, and evaluation benchmarks to run non-interactively.
- **Alternatives Considered:** Separate human-in-the-loop and automated runner classes. Rejected to keep the orchestration path identical between test and production environments.
- **Impact:** Ensures zero operational friction during development and automated evaluation while guaranteeing strict human oversight in production.

---

### Decision 7: Architecture for Episodic Memory Semantic Similarity Search (Planned in `genai-core`)
- **Context:** The requirements call for retrieving similar past plans from episodic memory. Direct inspection of `genai_core.EpisodicMemory` revealed that it currently supports canonical SHA-256 fingerprinting for idempotency checks and recency-based retrieval (`get_episodes`), but does not currently feature vector embedding similarity search. The user confirmed that semantic similarity search for episodic memory will be added directly to `genai-core`.
- **Decision:**
  1. Define a clean consumer interface in `PlanningMemoryManager.get_similar_past_plans(user_id, goal)`.
  2. For current Module 18 operations, log completed plans as `ActionEpisode`s (`action_type="plan_execution"`) and retrieve recent plan episodes for user-isolated history.
  3. When semantic similarity search (`search_episodes`) is introduced to `genai-core`, Module 18 will delegate directly to that API without requiring architectural refactoring in Module 18.
- **Alternatives Considered:** Implementing a temporary local vector store for episodic memory inside Module 18. Rejected because memory infrastructure belongs strictly in `packages/genai-core`, avoiding code duplication and architectural drift.
- **Impact:** Preserves clean package boundaries and sets up seamless future adoption when `genai-core` is enhanced.

---

### Decision 8: Fair Benchmark Evaluation Protocol against Module 17 Reactive Baseline
- **Context:** Evaluating whether planning improves execution requires a scientifically valid, fair comparison against the un-planned reactive baseline from Module 17 (`research_and_support_agent.BoundedAgent`).
- **Decision:** Standardize the evaluation benchmark (`scripts/run_eval.py`):
  1. Evaluate both systems on the exact same 10 curated benchmark goals spanning research, investigation, operations, and resilience.
  2. Enforce identical underlying models (`gemini-3.5-flash-lite`), equalized global resource budgets ($40k$ tokens, $0.25$ USD), and identical shared tool/DB state.
  3. Evaluate the final deliverables using the exact same objective model-based judge and criteria.
  4. Measure and compare Completion Rate (%), Steps/Turns, Total Cost ($ USD), Wall-Clock Latency (s), and Replanning Recovery Rate (%).
- **Alternatives Considered:** Comparing against arbitrary external benchmarks or evaluating Module 18 in isolation without a baseline. Rejected because without a standardized baseline, the quantitative benefits of explicit planning cannot be demonstrated.
- **Impact:** Delivers concrete, reproducible empirical evidence of the value of structured planning, parallel execution, and verification.

---

### Decision 9: Clean Protocol Separation for Future Modules 19 (Reflection) and 20 (Specialist Delegation)
- **Context:** Module 18 will be reused by Module 19 (plan reflection) and Module 20 (specialist agent delegation). Future requirements must be accommodated without prematurely implementing their logic.
- **Decision:** Introduce lightweight Python `Protocol` interfaces:
  1. `PlanReflector` protocol in `protocols/reflector.py`: Allows injecting a reflection/critique engine between plan generation and execution.
  2. `StepExecutor` protocol in `protocols/specialist.py`: Abstracts step execution. In Module 18, `Module17StepExecutor` implements this protocol; in Module 20, a `SpecialistAgentRouter` can implement it to route steps to domain-specific agents.
- **Alternatives Considered:** Hardcoding direct calls to `BoundedAgent` inside `DAGScheduler`. Rejected because it would require major refactoring when specialist agents are introduced in Module 20.
- **Impact:** Decouples orchestration from execution, ensuring Module 18 is 100% extensible for upcoming modules with zero architectural debt.

---

### Decision 10: Structural Reachability and Depth-First Search Cycle Detection Order
- **Context:** Milestone 1 introduces `PlanValidator` verifying DAG validity. A closed dependency loop (e.g. $A \to B \to A$) represents both a cycle and an absence of unblocked root steps.
- **Decision:** Execute 3-state Depth-First Search (`_UNVISITED`, `_VISITING`, `_VISITED`) cycle detection prior to root reachability verification.
- **Alternatives Considered:** Checking root reachability first (`any(not s.depends_on for s in steps)`). Rejected because it would raise a generic `UnreachablePlanError` on cyclic subgraphs instead of diagnosing the exact directed cycle path (e.g. `step_a -> step_b -> step_a`), depriving the automated repair loop and human reviewer of actionable feedback.
- **Impact:** Guarantees precise cycle tracing in error messages, optimizing the prompt repair loop when feeding validation failures back to `PlanRepairer`.

---

### Decision 11: Dynamic Capability Discovery via StepExecutor & Config-Gated Episodic Retrieval
- **Context:**
  1. The Planner LLM needs to know what tools and capabilities exist to assign appropriate `PlanStep.tool_name_hint` values. Hardcoding tool names in prompt strings tightly couples planning to Module 17 tools and breaks when Module 20 introduces multiple heterogeneous specialist agents.
  2. Semantic vector similarity search for episodic memory (`search_episodes`) is planned for future `genai-core` releases but is not currently available. We must define the workflow and consumer interface without waiting for or modifying `genai-core`.
- **Decision:**
  1. Add `get_available_capabilities() -> dict[str, str]` to the `StepExecutor` protocol. When `PlanGenerator` initializes, it queries the executor (or executor router) for available capabilities and injects the dynamic capability catalogue directly into `build_planning_prompt` and `build_repair_prompt`.
  2. Implement a config-based toggle `enable_episodic_similarity_search: bool = False` in `AssistantConfig`. When False (default), `PlanningMemoryManager.get_similar_past_plans` short-circuits gracefully, returning an empty list to avoid unnecessary recency queries or latency overhead (zero-shot planning). When True, it queries semantic search if an async search function is provided, falling back to recent `ActionEpisode`s filtered by `action_type="plan_execution"`. When `genai-core` adds semantic episodic search, only a configuration toggle change is required.
- **Alternatives Considered:**
  1. Manually maintaining tool name constants in Module 18: Rejected because it duplicates tool registries and violates open-closed extensibility for Module 20.
  2. Implementing local vector embeddings for episodes inside Module 18: Rejected because episodic vector indexing is an infrastructure concern owned strictly by `packages/genai-core`.
- **Impact:** Clean decoupling of planner prompting from underlying agent implementations, full forward compatibility with Module 20, zero wasted token spend on un-indexed episodic memory, and single-flag enablement when `genai-core` releases episodic vector search.

---

### Decision 12: Immutable Plan Versioning & Atomic Run Record Filesystem Persistence
- **Context:** Human plan reviews allow approval, modification, or rejection. When a plan is modified, predecessor plans must be preserved immutably for auditability, debugging, and comparative evaluation. In addition, automated test suites and benchmarks need a zero-friction bypass (`auto_approve_review=True`). Furthermore, run records must be written to disk reliably without race conditions or file corruption if a process terminates abruptly.
- **Decision:**
  1. In `HumanPlanReviewGateway`, human modifications increment plan `version = parent_version + 1`, record `parent_version`, set `created_by = "human_reviewer"`, and record operator `change_summary`. Rejections raise `PlanRejectedError` containing the reviewer ID and explicit critique.
  2. Implement `auto_approve_review` flag in `AssistantConfig`, allowing automated testing, CI pipelines, and benchmarks to bypass operator prompts while preserving `created_by="planner_llm"`.
  3. In `FileRunRecordStore`, implement atomic write operations via temporary files (`run_{run_id}.tmp`) followed by atomic rename (`replace`), preventing partial JSON writes from being read by concurrent processes or surviving crashes as corrupt files.
- **Alternatives Considered:**
  1. In-place plan mutation: Rejected because mutating plans destroys provenance and makes tracking human interventions impossible.
  2. External database (PostgreSQL/SQLite) for run records: Rejected because Module 18 specification requires lightweight, file-based persistence (`runs/run_<id>.json`), while the `RunRecordStore` protocol ensures future stores can be plugged in without changing domain logic.
- **Impact:** Complete provenance tracking across plan versions, clean CI test automation, and robust local filesystem persistence.


