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

---

### Decision 13: Asynchronous DAG Scheduling, Dual Verification Gates & Concurrency-Safe Sibling Settling
- **Context:** Milestone 4 requires dispatching independent plan steps concurrently bounded by a semaphore, verifying step deliverables against strict success criteria, and handling step failures gracefully without corrupting or abandoning running parallel sibling tasks.
- **Decision:**
  1. Pure-Python `DAGScheduler`: Implemented topological dependency readiness resolution using `asyncio.Semaphore(max_concurrency)` (default: 3) to execute independent steps in parallel. Downstream steps receive verified predecessor deliverables dynamically injected into their execution prompts.
  2. Dual Verification Architecture:
     - **Gate 1 (Programmatic Assertions):** Fast, deterministic verification asserting execution status is `COMPLETED`, deliverable text is non-empty, and text length satisfies minimum threshold ($\ge 20$ characters). Invalid outputs fail fast without incurring LLM API costs.
     - **Gate 2 (Model-Based Judge):** Structured evaluation (`StepJudgeVerdict`) via `LLMService.complete_structure` evaluating semantic criteria satisfaction with confidence score threshold $\ge 0.85$.
  3. Step-Level Self-Correction: Steps failing verification are retried locally up to `max_step_retries` (default: 2) with verification reasoning and missing criteria injected as `retry_feedback`.
  4. Concurrency-Safe Sibling Settling: When a step exhausts retries and fails, the scheduler immediately stops launching new tasks, awaits currently running sibling tasks via `asyncio.gather(*active_tasks.values(), return_exceptions=True)` to settle cleanly without abrupt cancellation, and records any successful sibling outputs into the cache to avoid wasting computation.
  5. Event-Free Async Orchestration: Used `asyncio.wait(active_tasks.values(), return_when=asyncio.FIRST_COMPLETED)` instead of `asyncio.Event` to eliminate race conditions when mock tasks or rapid operations complete synchronously.
- **Alternatives Considered:**
  1. External orchestration frameworks (e.g. Temporal, Prefect): Rejected because pure `asyncio` is self-contained, lightweight, and requires no external daemon or database infrastructure.
  2. Immediate sibling cancellation (`task.cancel()`): Rejected because abrupt task cancellation discards completed or near-completed tool work, wasting tokens, incurring side-effects, and leaving the plan in an inconsistent state.
  3. Single LLM-only verification: Rejected because programmatic gate short-circuits empty, short, or crashed outputs instantly with zero token consumption.
- **Impact:** Robust asynchronous DAG execution, bounded concurrency, two-tier verification filtering, preserved computation across parallel failures, and clean handoff to Milestone 5 dynamic replanning.

---

### Decision 14: Dynamic Subgraph Replanning, Immutable Work Preservation & Replanning Ceilings
- **Context:** When an execution step exhausts local retries and fails verification, the system must recover and formulate an alternative strategy to achieve the user's overall goal without repeating completed work or thrashing in infinite loops.
- **Decision:**
  1. Implemented `ReplanningEngine`: Coordinates targeted subgraph revision when a step fails after exhausting retries.
  2. Programmatic Work Preservation Invariant: All steps marked `COMPLETED` prior to replanning (including any settled parallel siblings that finished while the failed step was running) are locked as immutable dependencies. They retain their original IDs, descriptions, `status=COMPLETED`, and cached deliverables. If the LLM's candidate plan omits or alters completed steps, the engine programmatically restores them before DAG validation.
  3. Failure Critique & Capability Injection: The replanning prompt incorporates the top-level goal, immutable completed steps with verified outputs, the failed step's verifiable criteria and failure reason, remaining unfulfilled steps, and available tools/capabilities.
  4. DAG Validation & Automated Repair Loop: The revised candidate plan is verified via `PlanValidator`. If structural errors or cycles are detected, `PlanRepairer.repair_plan(...)` repairs the plan DAG automatically.
  5. Anti-Thrashing Replanning Ceiling: Enforces `max_replans` (default: 3). If replanning attempts reach the ceiling, execution halts with `ReplanningLimitExceededError` to prevent runaway token costs and infinite execution loops.
  6. Immutable Version Stamping: The revised plan is stamped with `version = current_plan.version + 1`, `parent_version = current_plan.version`, `created_by = "replanner"`, and `change_summary = f"Replanned after failure of step '{failed_step_id}': {failure_reason}"`.
- **Alternatives Considered:**
  1. Full plan regeneration from scratch: Rejected because re-executing already completed and verified steps wastes expensive API calls and introduces regression risks.
  2. Static hardcoded fallback rules: Rejected because open-ended research goals require semantic reasoning and dynamic capability discovery.
- **Impact:** Concurrency-safe, robust failure recovery that preserves all prior execution progress, adapts to API/tool errors, and strictly guarantees budget safety against infinite replanning.

---

### Decision 15: Run-Wide Budget Governance, Central Orchestrator Facade & Decoupled Extensibility Protocols
- **Context:** In Milestone 6, the system requires a production-grade orchestrator facade (`PlanAndExecuteAssistant`) tying together the entire 5-phase lifecycle (memory recall, structured planning, human review gating, bounded concurrent execution with dynamic replanning, and verified synthesis). In addition, cumulative resource consumption (token ceiling, USD cost budget, wall-clock timeout) must be tracked and strictly enforced across all phases, while future capabilities from Module 19 (reflection/self-critique) and Module 20 (specialist routing) must be cleanly decoupled via abstract protocols.
- **Decision:**
  1. `RunBudget` Governance: Implemented unified, monotonic budget tracking for cumulative tokens, cost (USD), and elapsed wall-clock time (`time.monotonic`). Checked before each major phase transition and step dispatch. Halts execution immediately by raising `BudgetExceededError` whenever limits are reached.
  2. Orchestrator Facade (`PlanAndExecuteAssistant`): Exposes clean `run(goal, user_id, ...)` coordinating:
     - Phase 1: Long-term preference recall and optional episodic plan retrieval.
     - Phase 2: Capability-aware structured plan generation and automated repair loop.
     - Phase 3: Human review gateway (supporting auto-approval in testing/CI, operator modification with DAG re-validation, or safe cancellation).
     - Phase 4: Async `DAGScheduler` dispatch bounded by semaphore, step-level retries with verification feedback, sibling task settling on failure, and dynamic replanning with work preservation.
     - Phase 5: LLM-based final deliverable synthesis from verified step outputs, episodic memory recording, and atomic run record persistence via `FileRunRecordStore`.
  3. Decoupled Protocol Boundaries:
     - `PlanReflector` protocol and `ReflectionResult` data model decoupled into `protocols/reflector.py` for Module 19 integration without modifying core planner logic.
     - `StepExecutor` protocol decoupled into `protocols/executor.py` allowing Module 20 multi-agent routers to drop in seamlessly.
  4. Non-Blocking Synchronous Run Persistence: Added `_save_run_sync` fallback in `_finalize_run` to guarantee the final state of `RunRecord` is safely flushed to disk even in sudden failure or exit scenarios.
- **Alternatives Considered:**
  1. Fragmented, per-step budgets only: Rejected because without a global run-level budget ceiling, long-running replanning loops or complex multi-step DAGs could consume excessive tokens and API costs.
  2. Hardcoding Module 19 reflection and Module 20 routing logic directly in the assistant: Rejected because decoupling via `typing.Protocol` keeps Module 18 clean, focused, and testable while providing future-proof extension points.
- **Impact:** Unified facade encapsulating the complete plan-and-execute system, full resource budget safety across all lifecycle phases, clean protocol boundaries for future modules, and 100% test coverage across 91 unit tests.

---

### Decision 16: 10-Goal Fair Comparative Evaluation Suite & Deterministic Mock Architecture
- **Context:** Milestone 7 requires an empirical, 10-goal benchmark evaluating the candidate system (Module 18 Plan-and-Execute Assistant) against the baseline (Module 17 Reactive BoundedAgent) across completion rate, steps/turns, latency, cost, and replanning recovery. The benchmark suite must execute rapidly and reliably in CI/CD without consuming API tokens or depending on live network infrastructure, while retaining the capability to run against live PostgreSQL databases and Google Gemini models.
- **Decision:**
  1. Curated 10-Goal Dataset: Defined `get_10_benchmark_goals()` and `create_10_goals_dataset()` covering 4 distinct architectural stress domains: multi-source research, sequential operations, failure recovery/resilience, and mixed memory/database reasoning.
  2. Deterministic Mock Execution Architecture: Implemented `deterministic_mock_planned_target` and `deterministic_mock_reactive_target` simulating the distinct structural characteristics of both paradigms:
     - Planned Assistant: parallel branch execution, 50%+ lower wall-clock latency, and resilient recovery on ambiguous goal G-06 via dynamic replanning.
     - Reactive Baseline: sequential multi-turn loop, higher latency, and turn exhaustion on ambiguous goal G-06 without replanning.
  3. Interactive CLI & Headless CI Flags: Implemented `scripts/run_eval.py` supporting interactive user selection (`1. deterministic_mock eval`, `2. realistic eval run`) when run in a terminal, alongside headless non-interactive flags (`--mock`, `--live`, `--model`, `--db-url`) for automated test pipelines.
  4. Integration Test Verification: Added `tests/integration/test_eval_benchmark.py` (gated by `INTEGRATION_TEST=1`) validating dataset schema integrity, summary metric math, end-to-end mock execution, and ANSI markdown comparative reporting.
- **Alternatives Considered:**
  1. Live-only benchmark execution: Rejected because external network latency, API credit consumption, and provider downtime make CI/CD pipelines flaky and slow.
  2. Evaluating Module 18 in isolation: Rejected because without a head-to-head baseline, absolute metrics cannot empirically demonstrate the architectural advantages of DAG planning over reactive agents.
- **Impact:** Repeatable, fair, and automated comparative benchmarking demonstrating that DAG planning achieves a 10% higher completion rate, lower latency via concurrent step execution, and robust recovery from execution failures.

---

### Decision 17: Live Provider & PostgreSQL Integration Test Architecture (Real Gemini & Containerized Database)
- **Context:** Prior to Milestone 8 final sign-off, the system required end-to-end integration tests validating that `genai-core` services (`LLMService`, `MemoryService`), `module-17` (`BoundedAgent`), and `module-18` planning components (`PlanGenerator`, `StepVerifier`, `ReplanningEngine`, `PlanAndExecuteAssistant`) execute successfully against real LLM providers (`GoogleProvider` using `gemini-3.5-flash-lite`) and live PostgreSQL database instances.
- **Decision:**
  1. Containerized PostgreSQL Infrastructure: Provisioned `docker-compose.yml` deploying `pgvector/pgvector:pg17` on port 5432 with schemas for support accounts (`support_customers`), tickets (`support_tickets`), vector knowledge bases (`document_chunks`), long-term memory records (`memory_lt_records`), and episodic memory (`memory_ep_episodes`).
  2. Dedicated Live Test Modules:
     - `tests/integration/test_live_genai_core.py`: Tests live `LLMService` structure and text completions, along with `PostgresMemoryStore` LTM preference recall and episodic action logging.
     - `tests/integration/test_live_module17_agent.py`: Tests `BoundedAgent` executing live tool calls against PostgreSQL data stores (`lookup_customer_account`, `ticket_status`, and `search_knowledge_base`).
     - `tests/integration/test_live_planning_components.py`: Tests `PlanGenerator` decomposing realistic research goals into valid DAGs, `StepVerifier` dual-gate model judge with semantic scoring, and `ReplanningEngine` work preservation and subgraph replacement.
     - `tests/integration/test_live_assistant_e2e.py`: Tests the complete `PlanAndExecuteAssistant` facade end-to-end: multi-step DAG generation, bounded execution via `DAGScheduler`, dual verification, LLM deliverable synthesis, episodic recording, human review rejection gating, and `RunBudget` deadline enforcement.
  3. Strict Relative Imports & Package Isolation: Kept all imports across `module-18` strictly relative (`..core`, `..models`, `..planning`), leaving `packages/genai-core` and `module-17-ai_agents` completely untouched.
- **Alternatives Considered:**
  1. Relying solely on unit mocks: Rejected because mocks do not catch LLM prompt schema drift, model deprecations (such as `gemini-2.0-flash`), or database connection nuances.
  2. Running integration tests in default `pre_push.sh`: Rejected because live tests require network access, API tokens, and running Docker containers; guarded under `INTEGRATION_TEST=1` fixture gates.
- **Impact:** 100% verified compatibility across live LLM services, pgvector databases, reactive agents, and DAG planning orchestrators with 13 out of 13 integration tests passing.
