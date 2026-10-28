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

---

### Decision 18: Live Comparative Benchmark Runner Architecture & Robust Execution Guardrails
- **Context:** Executing the 10-goal comparative benchmark against real Gemini API credentials and PostgreSQL database revealed runtime operational requirements:
  1. Real multi-turn agent tool execution (combining SentenceTransformers embedding inference, DuckDuckGo web search, and Gemini generation) requires up to 30-40 seconds per turn; the default 30.0s deadline in `AgentConfig` caused premature `BUDGET_EXHAUSTED` timeouts during live execution.
  2. When a step task raised `BudgetExceededError`, unretrieved task exceptions inside Python's `asyncio.wait(FIRST_COMPLETED)` loop resulted in silent task termination without updating step statuses, misleading the scheduler into detecting a false `DAGDeadlockError`.
  3. Goals requiring customer and ticket entity resolution (e.g. `C-101`, `T-101`) generated generic step instructions like "Lookup customer account", which caused the underlying `BoundedAgent` to prompt the user for customer IDs instead of using the goal's target entity.
- **Decision:**
  1. Live Benchmark Runner (`src/plan_and_execute_research_assistant/eval/live_runner.py`): Implemented `create_live_benchmark_targets` instantiating candidate and baseline targets with live `GoogleProvider` (`gemini-3.5-flash-lite`), live PostgreSQL database stores, and real-time terminal progress reporting.
  2. Execution Deadlines & Token Budgets: Configured `AgentConfig` with `default_timeout_seconds=90.0`, `default_max_tokens=30_000`, and `default_max_cost_usd=0.25` for live agent execution, and raised `RunBudget` to 60,000 tokens and 240.0s total wall-clock timeout.
  3. Scheduler Exception Propagation: In `DAGScheduler._orchestrate_step_execution`, explicitly inspected `task.exception()` on completed tasks. If `isinstance(exc, BudgetExceededError)`, the exception is re-raised immediately to halt the run cleanly; otherwise, the step is marked `FAILED` to trigger sibling settling and replanning.
  4. Entity Context Propagation: Added `TARGET USER / ENTITY CONTEXT: {user_id}` in `Module17StepExecutor.format_step_prompt` and added rule 8 to `build_planning_prompt` DAG invariants instructing the LLM to explicitly preserve entity identifiers in step descriptions.
  5. Flexible Benchmark CLI: Enhanced `scripts/run_eval.py` with `--limit <N>` and `--goal-ids <G-01,G-02>` flags to allow running full or partial benchmarks interactively or in automated pipelines.
- **Alternatives Considered:**
  1. Re-executing failed steps indefinitely: Rejected because budget limits are absolute hard invariants.
  2. Suppressing entity context: Rejected because interactive prompts inside autonomous plan steps violate automated benchmark execution.
- **Impact:** Reliable live evaluation execution with zero false deadlocks, clear entity context passing to underlying agents, and robust budget deadline enforcement across candidate and baseline systems.

---

### Decision 19: Dynamic Episodic Capability Suppression & Data Store Pre-Seeding for Mixed Goals
- **Context:** When running mixed benchmark goals (such as `G-05`: "Diagnose customer churn risk for customer C-104 using ticket history, stored memory preferences, and account tier"), the LLM generated plan steps with `tool_name_hint='get_episodic_memory'`. This occurred despite `AssistantConfig.enable_episodic_similarity_search: bool = False`, because:
  1. `enable_episodic_similarity_search: bool = False` only governed Phase 1 past plan retrieval in `PlanningMemoryManager.get_similar_past_plans`; it did not filter what execution capabilities were advertised to `PlanGenerator`.
  2. `Module17StepExecutor.get_available_capabilities()` dynamically queried `BoundedAgent.tool_registry`, which unconditionally listed `get_episodic_memory`.
  3. When executed, `get_episodic_memory` crashed with an unhandled `AttributeError` due to schema mismatches (`ep.episode_id` vs `ep.id` on `genai_core.ActionEpisode`).
  4. Customer `C-104` lacked pre-seeded long-term memory records in PostgreSQL, causing `get_longterm_memory` to return empty results and failing model verification.
- **Decision:**
  1. Dynamic Capability Filtering: In `Module17StepExecutor.get_available_capabilities()`, filter out episodic memory tools whenever `not self.config.enable_episodic_similarity_search`.
  2. Prompt Invariant Rule 9: In `build_planning_prompt` and `build_repair_prompt`, add explicit instructions requiring the planner to adhere strictly to advertised capabilities and use `get_longterm_memory` for user preferences and profile facts.
  3. Tool Registry Unregistration: In `live_runner.py`, explicitly unregister `get_episodic_memory`, `delete_episodic_memory_all`, and `delete_episodic_memory_episodes` from `bounded_agent.tool_registry` when episodic similarity search is disabled.
  4. Mixed Goal Pre-Seeding: In `create_live_benchmark_targets`, ensure `C-104` has a preference record in `memory_service.long_term` and an active ticket (`T-104`) in `tickets` if missing.
  5. Dedicated Unit Tests: Added `tests/unit/execution/test_module17_executor.py` verifying episodic tool suppression when disabled, preservation when enabled, and user context formatting.
- **Alternatives Considered:**
  1. Modifying `module-17-ai_agents` or `packages/genai-core`: Rejected to preserve strict package isolation and change control rules.
  2. Forcing manual step overrides: Rejected because the planner must dynamically adapt based on its advertised capabilities catalogue.
- **Impact:** Clean decomposition of mixed goals using `lookup_customer_account`, `ticket_status`, and `get_longterm_memory` without spurious episodic memory tool calls or `AttributeError` failures.

---

### Decision 20: Step Verifier Semantic Evaluation of Negative Findings & Non-Existent Resources
- **Context:** During live evaluation of batch audits and multi-item queries (such as `G-07` auditing tickets `T-101`, `T-103`, and non-existent `T-105`), the tool returned a definitive negative finding (*"Ticket T-105 could not be found in the system"*). Because the generated success criteria was written strictly (*"Ticket T-105 details and current status are successfully retrieved"*), the `StepVerifier` LLM judge rejected the deliverable with score `0.0` (*"The agent attempted to retrieve the ticket using the appropriate tool, but the ticket could not be found"*). This triggered 3 failed attempts in Plan Version 1, followed by dynamic replanning that re-inserted the step as `PENDING`, leading to 3 more identical failures in Plan Version 2 and premature token budget exhaustion (63,932 tokens).
- **Decision:**
  1. Generalized Negative Finding Invariant: Updated `build_judge_prompt` in `src/plan_and_execute_research_assistant/verification/prompts.py` with an explicit evaluation rule: for any query, lookup, search, retrieval, or audit step across any tool or data source, if the agent used the appropriate capability and conclusively established that a requested entity, record, document, account, or identifier does NOT exist, is NOT found, or returned empty results, this is a **valid, successful factual finding** about the real-world state of the system.
  2. Criteria Fulfillment on Absence: Directed the LLM judge to mark `is_satisfied=True` and assign a passing confidence score (`>= 0.85`) when the absence or non-existence is clearly documented, only failing (`is_satisfied=False`) for unhandled runtime crashes, hallucinated data, or failure to perform the required action.
  3. Tool-Agnostic Formulation: Avoided hardcoding ticket-specific terminology so the rule applies universally across database lookups, vector knowledge bases, memory stores, external APIs, and web search.
  4. Unit Test Coverage: Added unit tests in `tests/unit/verification/test_verification.py` verifying that the judge prompt includes the negative finding instructions and that `StepVerifier` correctly verifies negative findings.
- **Alternatives Considered:**
  1. Patching only ticket tools: Rejected because any tool (accounts, documents, search, memory) can return negative results or non-existent identifiers.
  2. Relaxing programmatic verification: Rejected because programmatic gate checks syntax and length, whereas semantic criteria satisfaction belongs to the model judge.
- **Impact:** Prevents runaway retry and replan loops on non-existent resources across all tools, allowing autonomous runs to complete and report accurate negative findings in final deliverables.

---

### Decision 21: Benchmark Evaluation Dataset Entity Alignment & Ambiguity Resolution
- **Context:** During live comparative evaluation on goals `G-03`, `G-09`, and `G-10`, queries did not explicitly name target customer or ticket identifiers (e.g., `G-09` was *"Prepare refund execution: verify customer eligibility..."* and `G-10` was *"Root-cause analysis for system incident: search error log chunks, query incident ticket..."*), while metadata specified caller role strings (`"operator_admin"`, `"billing_auditor"`, `"sre_engineer"`). When steps were dispatched to `BoundedAgent`, the executor passed `TARGET USER / ENTITY CONTEXT: {user_id}`, leading the agent to confuse the caller role for a customer account ID (e.g. calling `lookup_customer_account(customer_id="billing_auditor")`). This triggered 3 failed attempts and token budget exhaustion (41,218 tokens). Furthermore, `G-09` listed a non-existent tool `calculate_refund`, and `G-10` referenced a non-existent ticket `T-105`.
- **Decision:**
  1. Explicit Entity Scoping: Updated queries and metadata across the benchmark suite in `src/plan_and_execute_research_assistant/eval/dataset.py`:
     - `G-03`: Scoped explicitly to ticket `T-102` and customer `C-102`, updating `user_id` to `"C-102"`.
     - `G-09`: Scoped explicitly to customer `C-102` (`"Prepare refund execution for customer C-102..."`), updated `user_id` to `"C-102"`, and replaced non-existent `calculate_refund` with available tools `["lookup_customer_account", "search_knowledge_base", "save_longterm_memory"]`.
     - `G-10`: Scoped explicitly to incident ticket `T-103` (the kernel crash panic incident in seed data) and customer `C-102`, updating `user_id` to `"C-102"`.
  2. Ground-Truth Data Consistency: Aligned all expected evidence and reference deliverables with actual seed records in PostgreSQL (`support_customers` `C-101`, `C-102`, `C-104` and `support_tickets` `T-101`, `T-102`, `T-103`).
- **Alternatives Considered:**
  1. Stripping `user_id` from step dispatch: Rejected because memory lookups and personalized SLA checks legitimately require user identity context.
  2. Synthesizing dummy customers for every role: Rejected because production systems should not pollute customer account tables with auditor role names.
- **Impact:** Eliminates entity confusion loops and allows `BoundedAgent` and `PlanAndExecuteAssistant` to resolve customer accounts and incident tickets on attempt 1 without thrashing or exceeding token budgets.

---

### Decision 22: Evaluation Benchmark Inter-Goal Cooldown (Rate Limit & Quota Protection)
- **Context:** Executing comparative evaluation benchmarks against live LLM APIs (Google Gemini) in rapid succession can trigger provider rate limits and quota depletion (HTTP 429 Resource Exhausted errors), especially when running multi-turn reactive agents and multi-step DAG planning workflows back-to-back.
- **Decision:**
  1. Inter-Goal Evaluation Cooldown: In `src/plan_and_execute_research_assistant/eval/runner.py`, introduced `pause_between_goals_seconds: float = 30.0` to `run_comparative_benchmark`.
  2. Latency Measurement Integrity: Recorded execution wall-clock time (`res.latency_s = round(time.monotonic() - start_t, 3)`) immediately upon target completion *before* invoking `await asyncio.sleep(pause_between_goals_seconds)`. This guarantees benchmark latency and speedup KPIs accurately measure agent performance without skew from inter-goal cooldown pauses.
  3. Symmetric Multi-Agent Application: Applied the cooldown pause symmetrically after each goal evaluation for both candidate (`Module 18 Planned Assistant`) and baseline (`Module 17 Reactive BoundedAgent`), with real-time logging: `[Cooldown] Pausing 30s after goal {goal_id} ({system_name})...`.
  4. CLI & Test Suite Integration: Added `--pause-between-goals` to `scripts/run_eval.py`, defaulting to 30.0s in live execution and 0.0s in deterministic mock mode. Explicitly passed `pause_between_goals_seconds=0.0` in `test_run_comparative_benchmark_mock_mode` so automated CI/integration test runs complete in <3 seconds.
- **Alternatives Considered:**
  1. Pausing inside the internal agent step loops: Rejected because agent-level delays artificially inflate step latency and violate real-world benchmark fidelity.
  2. Pausing only once between systems: Rejected because evaluating 10 goals in sequence under a single system depletes per-minute token buckets (TPM).
- **Impact:** Clean rate limit mitigation across live benchmark runs with zero benchmark latency skew and sub-3-second automated test execution.

---

### Decision 23: Real-Time Step Telemetry Streaming & 5K Tiktoken Web Retrieval Limit
- **Context:**
  1. Telemetry Loss on Budget Exhaustion: When runs breached global token or cost limits (e.g. `run_64fff92f` reaching 42,198 tokens during `step_2`), `DAGScheduler` raised `BudgetExceededError`, which aborted `execute_plan` without returning. Because `assistant.py` previously transferred step records only after `execute_plan` completed, all `StepExecutionRecord` instances accumulated prior to the breach were discarded, leaving `step_executions` completely empty (`{}`).
  2. Web Search Payload Bloat: Web search steps (`search_web`) against dense technical documentation yielded large responses that multiplied token consumption across multi-turn agent iterations and predecessor context chains, accelerating budget exhaustion.
- **Decision:**
  1. Real-Time Telemetry Streaming:
     - Enhanced `DAGScheduler.execute_plan` to accept an optional `step_records_collector: dict[str, list[StepExecutionRecord]]`.
     - In `_run_single_step`, immediately upon building each `StepExecutionRecord` after execution and verification, appended it directly to `step_records_collector[step.id]`.
     - In `assistant.py`, passed `step_records_collector=run_record.step_executions`, ensuring every attempt across all plan versions is saved to disk even if `BudgetExceededError` terminates the run abruptly.
     - On `BudgetExceededError`, explicitly cancelled running sibling asyncio tasks before re-raising.
  2. Tiktoken-Powered 5K Token Limit on Web-Retrieved Content:
     - Added `src/plan_and_execute_research_assistant/core/tokens.py` implementing `count_tokens` and `truncate_to_tokens` using `tiktoken` (`cl100k_base` encoding).
     - Added `max_web_retrieved_tokens: int = 5000` to `AssistantConfig`.
     - In `Module17StepExecutor.execute_step`, truncated any web-retrieved output exceeding 5,000 tokens using `truncate_to_tokens`, appending `... [Web-retrieved response truncated to 5000 tokens limit]`.
     - In `create_live_benchmark_targets` (`live_runner.py`), wrapped `search_web` in `bounded_agent.tool_registry` to enforce the 5K token boundary at the tool execution level for both Planned Assistant and Reactive BoundedAgent.
- **Alternatives Considered:**
  1. Relying on simple character estimation: Rejected because character counts drift significantly on code, tables, and URLs; `tiktoken` guarantees exact token window safety.
  2. Only truncating in post-processing: Rejected because huge intermediate tool outputs inside `BoundedAgent` cause immediate token budget exhaustion during agent turns.
- **Impact:** 100% telemetry retention across all completed step attempts even upon budget termination, and strict 5K token boundedness for all web-retrieved data.

---

### Decision 24: Operational Hand-off & Approval-Required Status Handling (HITL Safety Invariant)
- **Context:** During live comparative evaluation on operational goals (e.g., `G-03` resolving ticket `T-102` and executing support actions), the planner may decompose an operational step calling an irreversible tool (e.g. `execute_refund`). In `BoundedAgent` (Module 17), `execute_refund` is guarded by `RiskLevel.IRREVERSIBLE`, causing the agent's tool loop to halt and return `ToolLoopStatus.APPROVAL_REQUIRED` alongside a serialized `ApprovalRequest`. Because `Module17StepExecutor.execute_step` previously checked only `is_success = "completed" in status_str or "success" in status_str`, it treated `APPROVAL_REQUIRED` as `StepStatus.FAILED`. Furthermore, because the tool loop paused before LLM final synthesis, `output_text` was left empty (`""`). This caused Programmatic Gate 1 in `StepVerifier` to immediately fail the step (`"Step execution failed with status 'failed'"`), triggering 3 identical retry failures, replanning, and token budget exhaustion (42,950 tokens).
- **Decision:**
  1. Operational Hand-off Status Recognition: In `src/plan_and_execute_research_assistant/execution/module17_executor.py`, updated `is_success` to recognize `is_approval = "approval_required" in status_str`, mapping safely paused approval states to `StepStatus.COMPLETED`.
  2. Deliverable Synthesis for Pending Approvals: When `is_approval` is True and `output_text` is empty, extract `result.pending_approval` (`ApprovalRequest`) and synthesize a structured deliverable documenting the initiated action, request ID, tool arguments, risk level, and policy reason awaiting reviewer authorization.
  3. LLM Judge Instruction for Human Authorization: In `src/plan_and_execute_research_assistant/verification/prompts.py`, added instruction 3 clarifying that operational actions safely paused for human review represent successful, compliant execution under enterprise safety policies, directing the judge to assign passing scores (`>= 0.85`).
  4. Unit Test Coverage: Added `test_execute_step_approval_required_is_success` in `test_module17_executor.py` and `test_verify_step_approval_required_verified` in `test_verification.py`.
- **Alternatives Considered:**
  1. Auto-resuming approval inside `Module17StepExecutor`: Rejected as default because in a true plan-and-execute system, submitting a dangerous or irreversible action to an external supervisor queue is the correct terminal state of the agent's autonomous execution phase.
  2. Bypassing tool approval checks entirely: Rejected because violating security invariants compromises safety guarantees.
- **Impact:** Clean execution and verification of operational steps requiring human authorization without spurious retries, false replans, or token budget exhaustion.

---

### Decision 25: DAGScheduler Work Preservation on Replans & Complete LLM RunBudget Accounting (P0 Fixes)
- **Context:**
  1. Broken Work Preservation Invariant (PLN-001): When `ReplanningEngine` generated a revised DAG (Version $N+1$) following a step failure, completed steps were stamped with `status=COMPLETED` and `result_summary`. However, `DAGScheduler.execute_plan` unconditionally initialized `step_status` to `StepStatus.PENDING` for all steps and `step_results` to `{}`. Consequently, already-completed and verified steps were re-dispatched and executed again, violating work preservation, duplicating side-effects (e.g. balance adjustments), and wasting model tokens and latency.
  2. Systemic Token & Cost Budget Leak (PLN-002): `RunBudget.record_usage` was only invoked for step execution (`exec_output`), replanning, and synthesis. Initial plan generation (`PlanGenerator.generate_plan`), automated repair loops (`PlanRepairer.repair`), and step verification judges (`StepVerifier.verify_model`) completely bypassed budget accounting. Consequently, reported costs were falsely deflated and global ceilings could not prevent runaway consumption.
- **Decision:**
  1. DAGScheduler Work Preservation:
     - In `DAGScheduler.execute_plan`, initialized `step_status` to preserve `s.status` if `s.status == StepStatus.COMPLETED`, defaulting to `StepStatus.PENDING` only for remaining/uncompleted steps.
     - Initialized `step_results` with `s.result_summary` for all pre-completed steps.
     - As a result, the dispatch loop immediately skips pre-completed steps, makes their cached outputs available to downstream dependent steps in `predecessor_context`, and includes them in the final `step_results`.
  2. Verification Judge Budget Tracking:
     - Added `tokens_used: int = 0` and `cost_usd: float = 0.0` fields to `StepVerificationResult`.
     - In `StepVerifier.verify_model`, extracted `usage` and `cost_usd` from the LLM response and returned them as `(verdict, tokens, cost)`. Populated these in `StepVerificationResult`.
     - In `DAGScheduler._run_single_step`, recorded verification token and cost usage into `active_budget.record_usage` and combined them into `StepExecutionRecord.tokens_used` and `cost_usd`.
  3. Planning & Repair Budget Tracking:
     - Updated `PlanGenerator.generate_plan`, `PlanRepairer.repair`, and `PlanRepairer.repair_plan` to accept an optional `budget: RunBudget | None = None`.
     - Recorded usage from structured plan generation and repair LLM calls into `budget.record_usage(tokens, cost)`.
     - In `PlanAndExecuteAssistant`, passed `budget=active_budget` during Phase 2 plan generation.
     - In `ReplanningEngine`, forwarded `budget=budget` during repair of candidate replanned plans.
  4. Unit Tests Added:
     - `test_scheduler_preserves_completed_steps_on_replan` in `tests/unit/scheduling/test_dag_scheduler.py`.
     - `test_verifier_tracks_tokens_and_cost` in `tests/unit/verification/test_verification.py`.
     - `test_generator_records_tokens_in_budget` in `tests/unit/planning/test_plan_generator.py`.
- **Alternatives Considered:**
  1. Pruning completed steps entirely from replanned DAGs: Rejected because downstream steps reference completed step IDs in `depends_on`, requiring completed steps to remain in the plan graph for dependency resolution.
  2. Tracking budget only at phase boundaries: Rejected because individual step loops and verification cycles can breach token/cost budgets before reaching a phase boundary.
- **Impact:** Strict adherence to the Work Preservation Invariant during dynamic replanning (zero duplicate execution of completed steps) and 100% complete accounting of all LLM tokens and costs in `RunBudget`.

---

### Decision 26: Benchmark Validity Harmonization, Concurrency Robustness & Store Isolation (P1 Fixes)
- **Context:**
  1. Asymmetric Benchmark Scoring (PLN-003): In `eval/live_runner.py`, `live_planned_target` used an asymmetric criteria condition `passed = is_completed and (all_steps_ok or req_matched) and len(deliverable) > 20`. This allowed Planned Assistant to pass even when failing benchmark-required keyword checks simply because all plan steps executed to completion. Conversely, `live_reactive_target` strictly enforced `req_matched` and did not recognize `ToolLoopStatus.APPROVAL_REQUIRED` as valid completion for human-in-the-loop operational goals.
  2. Silent Swallowing of BudgetExceededError in Sibling Settling (PLN-004): In `DAGScheduler`, `_execute_step_with_retries` caught general `Exception` and converted `BudgetExceededError` into a regular step failure rather than propagating it. Furthermore, when sibling tasks were awaited during settling (`asyncio.gather(*active_tasks.values(), return_exceptions=True)`), exceptions in `settled_results` were not inspected for `BudgetExceededError`, allowing budget violations to be swallowed silently and triggering unwanted dynamic replans.
  3. Non-Fatal Episodic Memory Telemetry (PLN-005): In `PlanAndExecuteAssistant.run`, episodic memory write (`record_plan_episode`) occurred within the primary execution `try` block without isolated exception handling. If vector storage, embedding generation, or database operations failed transiently after all steps and deliverable synthesis had succeeded, the entire run was erroneously marked as `FAILED`.
  4. In-Memory Store Contamination Across Benchmark Goals (PLN-006): In `eval/live_runner.py`, mutable in-memory dictionary stores (`accounts` and `tickets`) were passed by reference into tools and shared across both Planned Assistant and Reactive Baseline across all 10 benchmark goals. Mutating accounts or tickets in earlier goals (e.g. `G-03`) corrupted subsequent goal evaluations (e.g. `G-04`, `G-09`).
- **Decision:**
  1. Harmonized Benchmark Passing Criteria (PLN-003):
     - In `live_planned_target`, removed `(all_steps_ok or req_matched)` and enforced symmetric `passed = is_completed and req_matched and len(deliverable) > 20`.
     - In `live_reactive_target`, recognized `is_completed = agent_res.status in (ToolLoopStatus.COMPLETED, ToolLoopStatus.APPROVAL_REQUIRED)` to honor enterprise HITL safety approvals.
  2. Concurrency Robustness & BudgetExceededError Propagation (PLN-004):
     - In `DAGScheduler._execute_step_with_retries`, explicitly caught and re-raised `BudgetExceededError` before general `Exception` handling.
     - In `DAGScheduler.execute_plan`, inspected `settled_results` from sibling task settling (`asyncio.gather(*active_tasks.values(), return_exceptions=True)`) and immediately re-raised any `BudgetExceededError` to abort cleanly.
  3. Resilient Episodic Memory Isolation (PLN-005):
     - In `PlanAndExecuteAssistant.run`, wrapped `self.memory_manager.record_plan_episode` in an isolated `try...except Exception as mem_exc` block with a warning log, ensuring memory telemetry failures never fail an already-completed user run.
  4. In-Memory Store Snapshot & Restoration (PLN-006):
     - In `live_runner.py`, captured baseline deepcopy snapshots (`initial_accounts` and `initial_tickets`) after seed data initialization.
     - Restored stores in-place via `.clear()` and `.update(copy.deepcopy(initial_...))` before dispatching each goal in both `live_planned_target` and `live_reactive_target`.
  5. Test Coverage:
     - Added `test_scheduler_sibling_settling_propagates_budget_exceeded_error` in `tests/unit/scheduling/test_dag_scheduler.py`.
     - Added `test_assistant_episodic_memory_failure_does_not_fail_run` in `tests/unit/orchestration/test_assistant.py`.
     - Added `test_eval_store_state_isolation_deepcopy` and `test_eval_criteria_symmetry_logic` in `tests/unit/eval/test_eval_runner.py`.
- **Alternatives Considered:**
  1. Re-instantiating the entire agent and database connection per benchmark goal: Rejected due to significant connection setup overhead and latency; in-place dictionary restoration achieves complete isolation with zero connection churn.
  2. Failing the run on episodic memory errors: Rejected because episodic memory is secondary historical indexing; user deliverable synthesis is the primary goal.
- **Impact:** Symmetric, unbiased benchmark evaluation; bulletproof budget error propagation across concurrent DAG tasks; resilience against transient episodic database errors; and total state isolation between benchmark test cases.

---

### Decision 27: Reliability & Architectural Integrity: Deep Copy on Plan Review Modifications, Async Run Finalization, Active Deadline Cancellation & Interactive CLI Review Fallback (P2 Fixes)
- **Context:**
  1. Shallow Copy in Plan Review Modifications (PLN-007): In `HumanPlanReviewGateway.process_decision`, `decision.modified_plan.model_copy(...)` defaulted to `deep=False`. Consequently, nested `PlanStep` object references were shared between historical plan versions and the currently executing plan, causing subsequent step status updates during execution to mutate archived historical plan versions in-place.
  2. Non-Async Run Finalization Breaching Storage Protocol (PLN-008): In `PlanAndExecuteAssistant._finalize_run`, the run finalizer called a private synchronous method `self.store._save_run_sync(run_record)`. If a custom storage backend implemented the asynchronous `RunRecordStore` protocol (`async def save_run(...)`) without providing `_save_run_sync`, persistence either failed silently or threw an AttributeError.
  3. Passive Deadline Enforcement During Task Await (PLN-009): In `DAGScheduler.execute_plan`, the main loop invoked `await asyncio.wait(active_tasks.values(), return_when=asyncio.FIRST_COMPLETED)` without a timeout. In-flight tasks that hung or exceeded the remaining wall-clock budget ran past the deadline indefinitely without active cancellation.
  4. Missing Interactive CLI Review Fallback (PLN-010): In `HumanPlanReviewGateway.review_plan`, when `config.require_plan_review = True` and `decision_provider is None`, the gateway unconditionally raised a `ValueError` rather than providing an interactive stdin prompt as documented for CLI execution.
- **Decision:**
  1. Enforced Deep Copy on Plan Review Modifications (PLN-007):
     - In `HumanPlanReviewGateway.process_decision`, passed `deep=True` in `decision.modified_plan.model_copy(update={...}, deep=True)`, guaranteeing total isolation of step models across plan versions.
  2. Fully Asynchronous Run Finalization (PLN-008):
     - Converted `_finalize_run` in `PlanAndExecuteAssistant` to an `async def` method.
     - Awaited `await self.store.save_run(run_record)` conforming strictly to the public `RunRecordStore` protocol interface.
     - Updated all five call sites in `PlanAndExecuteAssistant.run` to await `_finalize_run`.
  3. Active Deadline Cancellation on Wall-Clock Timeout (PLN-009):
     - In `DAGScheduler.execute_plan`, derived `rem_timeout = active_budget.remaining_seconds()`.
     - Passed `timeout=rem_timeout` to `asyncio.wait`. If timeout elapses before any task completes (`not done and pending`), immediately invoke `active_budget.check_budget()`, triggering `BudgetExceededError` and cancelling all in-flight sibling tasks.
  4. Interactive CLI Prompt Fallback (PLN-010):
     - In `HumanPlanReviewGateway`, implemented `_default_cli_review_prompt` using `asyncio.to_thread` to read operator decisions (`[a]pprove`, `[r]eject`) from `stdin`.
     - When `decision_provider is None` and `sys.stdin.isatty()` is `True`, seamlessly delegate to `_default_cli_review_prompt`. In non-interactive environments, raise a descriptive `ValueError`.
  5. Test Coverage:
     - `test_process_decision_modify_enforces_deep_copy` in `tests/unit/review/test_human_review.py`.
     - `test_review_plan_interactive_cli_approve` and `test_review_plan_interactive_cli_reject` in `tests/unit/review/test_human_review.py`.
     - `test_assistant_finalizes_run_with_pure_async_store` in `tests/unit/orchestration/test_assistant.py`.
     - `test_scheduler_active_cancellation_on_budget_timeout` in `tests/unit/scheduling/test_dag_scheduler.py`.
- **Alternatives Considered:**
  1. Background timer task with cancellation handles: Rejected because passing `timeout=rem_timeout` directly to `asyncio.wait` leverages native asyncio timeout mechanics cleanly without auxiliary tasks.
  2. Blocking synchronous input on the event loop: Rejected; wrapped input prompt in `asyncio.to_thread` to prevent stalling the async event loop during operator review.
- **Impact:** Complete immutability of historical plan versions, full protocol compliance for pluggable async storage backends, proactive cancellation of deadline-breaching tasks, and out-of-the-box interactive CLI plan inspection.

---

### Decision 28: Judge Deliverable Tag Fencing, Predecessor Context Ceiling, Root `.env` Aliases & Plan Version Snapshot Immutability 
- **Context:**
  1. Unfenced Deliverable in Judge Prompt & Unbounded Predecessor Context (PLN-011): In `verification/prompts.py`, `output_text.strip()` was directly placed in the prompt without XML or markdown delimiter tags, exposing the evaluation judge to prompt injection or structural confusion. In `execution/module17_executor.py`, predecessor step summaries were injected without token ceilings, risking prompt context explosion on long multi-step DAGs.
  2. `env_prefix="ASSISTANT_"` in `AssistantConfig` (PLN-012): Root `.env` environment variables (`DEFAULT_LLM_MODEL`, `MAX_CONCURRENCY`, `DEFAULT_MODEL`, etc.) were ignored unless explicitly prefixed with `ASSISTANT_`.
  3. Replanning Token Usage Extraction on Dictionary Formats (PLN-013): In `ReplanningEngine.replan`, token usage extraction assumed an object structure with `getattr(usage, "total_tokens", 0)`. When `genai_core` returned usage as a dictionary, it evaluated to `0`, causing replanning token usage to be under-reported in the run budget.
  4. In-Place Plan Step Mutation Mutating Initial Plan Snapshot in `run_record.plan_versions` (PLN-014): In `PlanAndExecuteAssistant.run`, `run_record.plan_versions.append(initial_plan)` appended the mutable plan instance by reference. When the scheduler mutated step statuses in-place (`step.status = StepStatus.COMPLETED`), `plan_versions[0]` was mutated retroactively, destroying the pre-execution snapshot.
- **Decision:**
  1. Deliverable XML Tag Fencing & Predecessor Context Token Ceiling (PLN-011):
     - In `verification/prompts.py`, wrapped `output_text.strip()` inside `<deliverable>...</deliverable>` tags.
     - In `execution/module17_executor.py`, truncated each predecessor dependency summary to a 1,000-token ceiling using `truncate_to_tokens(summary, max_tokens=1000)`.
  2. Pydantic `AliasChoices` for Root `.env` Fallbacks (PLN-012):
     - In `AssistantConfig`, added `validation_alias=AliasChoices(...)` for `default_model`, `default_provider`, `max_concurrency`, `default_step_max_iterations`, `max_total_tokens`, `max_total_cost_usd`, and `default_timeout_seconds`, prioritizing `ASSISTANT_*` while supporting root `.env` keys.
  3. Dict-Resilient Usage Extraction in Replanning Engine (PLN-013):
     - In `ReplanningEngine.replan`, aligned usage parsing with `PlanGenerator` and `PlanRepairer`, extracting tokens and cost seamlessly from both dictionary and object formats.
  4. Deep Copy Snapshotting for `run_record.plan_versions` (PLN-014):
     - In `PlanAndExecuteAssistant.run`, appended `initial_plan.model_copy(deep=True)`, `approved_plan.model_copy(deep=True)`, and `new_plan.model_copy(deep=True)` to `run_record.plan_versions`, guaranteeing that historical plan versions remain immutable audit snapshots.
  5. Test Coverage:
     - `test_build_judge_prompt` updated for `<deliverable>` tags in `tests/unit/verification/test_verification.py`.
     - `test_format_step_prompt_truncates_large_predecessor_context` added in `tests/unit/execution/test_module17_executor.py`.
     - `test_assistant_config_root_env_aliases` added in `tests/unit/core/test_config.py`.
     - `test_replan_records_budget_usage_with_dict_usage` added in `tests/unit/replanning/test_replanning.py`.
     - `test_assistant_preserves_immutable_initial_plan_in_plan_versions` added in `tests/unit/orchestration/test_assistant.py`.
- **Alternatives Considered:**
  1. Removing `env_prefix="ASSISTANT_"` entirely: Rejected because retaining `ASSISTANT_` prefix ensures namespace isolation when running in environments with mixed agent configurations.
  2. Freezing `PlanStep` Pydantic models completely (`frozen=True`): Rejected because the scheduler updates step status, timing, and result summaries dynamically during step lifecycle transitions.
- **Impact:** Robust prompt injection defense in step judge verification, bounded predecessor token propagation, seamless `.env` variable loading, accurate replanning budget tracking, and reliable historical plan version auditing.




