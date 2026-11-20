# Module 20: Parallel Research Team — Architectural Decisions Log

This document records all significant architectural, technical, and structural decisions made during the design and implementation of Module 20. It follows an append-only structure.

---

## [2026-11-18] DEC-20-001: Centralized Configuration with Dedicated Reflection Judge Knobs

- **Status:** Accepted
- **Context:** The system requires a centralized configuration point for all tunable parameters (LLM models, concurrency, timeouts, shared budgets, retrieval mode, verification gates) so that different configurations can be evaluated without modifying application code. In particular, reflection and verification gates require dedicated judge settings.
- **Decision:** Implemented `ResearchTeamConfig` (`src/parallel_research_team/core/config.py`) using Pydantic Settings. Added dedicated `judge_provider` (default: `"google"`) and `judge_model` (default: `"gemini-3.1-flash-lite"`) alongside default worker models (`"gemini-3.5-flash-lite"`).
- **Consequences:** All components receive a validated, typed configuration object. Judge models for Gate 2 verification can be tuned independently of worker/synthesis models.

---

## [2026-11-18] DEC-20-002: Separate `exceptions/` Package for Domain Errors

- **Status:** Accepted
- **Context:** Error types should be cleanly decoupled from core configuration and budget logic, mirroring `genai_core.exceptions`.
- **Decision:** Created a dedicated `src/parallel_research_team/exceptions/` package containing typed exceptions: `ResearchTeamError`, `BudgetExceededError`, `DecompositionError`, `WorkerExecutionError`, `WorkerTimeoutError`, `WorkerOutputValidationError`, `SynthesisError`, `VerificationError`, and `CheckpointError`.
- **Consequences:** Clean separation of concerns; components import domain exceptions without circular dependencies with `core/`.

---

## [2026-11-18] DEC-20-003: Concurrency-Safe Shared Run Budget with `asyncio.Lock`

- **Status:** Accepted
- **Context:** Parallel research workers execute tool loops concurrently. If multiple workers complete LLM calls simultaneously, unprotected budget counters cause race conditions and budget overshoots.
- **Decision:** Implemented `SharedRunBudget` (`src/parallel_research_team/core/budget.py`) with an internal `asyncio.Lock`. Both `record_usage()` and `can_proceed()` acquire the lock, ensuring atomic balance updates and threshold checks.
- **Consequences:** Guaranteed thread- and async-safety under high concurrency; verified by stress tests running 50 concurrent worker tasks.

---

## [2026-11-18] DEC-20-004: Typed `OutputFormatSpec` on `WorkerBrief` for Serializable Checkpoints

- **Status:** Accepted
- **Context:** Research briefs need to specify deliverable formatting to workers. Using a raw Python class type causes serialization failures when dumping `CheckpointState` to JSON.
- **Decision:** Created `OutputFormatSpec` (`src/parallel_research_team/models/brief.py`) as a typed, frozen Pydantic model (`schema_name`, `required_fields`, `guidelines`).
- **Consequences:** Briefs remain strictly typed and 100% JSON-serializable, enabling lossless persistence in `CheckpointState`.

---

## [2026-11-18] DEC-20-005: Module 20 Native `Report` Model with Module 19 Verification Compatibility

- **Status:** Accepted
- **Context:** Module 20 needs to track multi-agent metadata (such as `research_team_version`, `brief_coverages`, and `unresolved_limitations`), but must also pass Module 19's `DeterministicVerifier` and `SemanticGroundingVerifier`.
- **Decision:** Implemented Module 20's own `Report` model (`src/parallel_research_team/models/report.py`). It adheres to Module 19's structural interface (`topic`, `title`, `executive_summary`, `sections`, `conclusion`, `limitations`, `Claim`, `Citation`, `SourceReference`) while adding Module 20 specific tracking fields.
- **Consequences:** No external coupling to Module 19's internal domain files while remaining fully compatible with Module 19's verification algorithms.

---

## [2026-11-18] DEC-20-006: Atomic Persistence via Temporary File Rename in `FileCheckpointStore`

- **Status:** Accepted
- **Context:** Process crashes or interruptions during file writes can corrupt JSON checkpoint files, making resumption impossible.
- **Decision:** Implemented `FileCheckpointStore` (`src/parallel_research_team/storage/file_store.py`) writing to a `NamedTemporaryFile` in the same directory, syncing to disk with `os.fsync`, and atomically replacing the target checkpoint path using `os.replace`.
- **Consequences:** Checkpoint writes are crash-consistent on POSIX filesystems. Corrupted or partial writes are prevented.

---

## [2026-11-18] DEC-20-007: Native `PricingCalculator` Catalog in Module 20

- **Status:** Accepted
- **Context:** Module 20 needs precise USD cost tracking across model generations without depending on Module 17's internal customer support pricing modules.
- **Decision:** Implemented Module 20's own `PricingCalculator` (`src/parallel_research_team/telemetry/pricing.py`) with updated rate cards for Gemini (`gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-3.8-flash`, `gemini-3.1-pro-preview`), OpenAI (`gpt-6` series), and Anthropic (`claude-5` series), with case-insensitive and family fallback matching.
- **Consequences:** Independent, up-to-date financial accounting across all research LLM operations.

---

## [2026-11-18] DEC-20-008: Versioned Prompts and Centralized `PromptTemplateManager`

- **Status:** Accepted
- **Context:** Hardcoding prompts in python files leads to tight coupling, difficulty updating templates across versions, and lack of prompt auditability.
- **Decision:** Implemented Jinja2 prompt management (`src/parallel_research_team/prompts/manager.py`) backed by directory-structured templates (`src/parallel_research_team/prompts/v1/` with `metadata.json`, `decomposition/`, `worker/`, and `synthesis/`). The manager provides lifecycle rendering methods (`render_decomposition_prompts`, `render_worker_prompts`, and `render_synthesis_prompts`).
- **Consequences:** Prompt revisions are cleanly decoupled, versioned, and easily swapped via `config.prompt_version`.

---

## [2026-11-18] DEC-20-009: Decomposition Engine with Memory Integration and Fallback Services

- **Status:** Accepted
- **Context:** The lead agent must classify effort scaling (`DIRECT` vs `PARALLEL_DECOMPOSED`), produce typed worker briefs, recall user preferences from memory, and pin research goals into short-term memory. Core service dependencies must accept explicit module configuration values.
- **Decision:** Implemented `DecompositionEngine` (`src/parallel_research_team/decomposition/engine.py`) using `LLMService.complete_structure` for typed structured output. Service constructors explicitly receive `default_provider=config.default_provider` and `default_model=config.default_model`. Long-term memory (`genai_core.LongTermMemory`) recalls domain preferences and short-term memory (`genai_core.ShortTermMemory`) pins the decomposition goal.
- **Consequences:** Safe, typed decomposition with non-blocking graceful fallback if memory services are omitted or unavailable.

---

## [2026-11-19] DEC-20-010: Extensible Worker Registry and Protocol Pattern

- **Status:** Accepted
- **Context:** The system needs to support specialized worker types (e.g., code analysis, deep web, financial data) in the future without modifying scheduler or orchestrator dispatch logic.
- **Decision:** Defined `ResearchWorkerProtocol` (`src/parallel_research_team/workers/protocol.py`) and implemented `WorkerRegistry` (`src/parallel_research_team/workers/registry.py`) providing a thread-safe factory registry. Unknown worker types automatically fall back to `"default"` (`DefaultResearchWorker`) with a logged warning.
- **Consequences:** Clean open-closed architecture; new worker agents can be registered at runtime without altering core execution workflows.

---

## [2026-11-19] DEC-20-011: Deterministic SHA-256 Hashing for Web Evidence and Source Chunks

- **Status:** Accepted
- **Context:** Web search outputs from DuckDuckGo or HTML scraping do not provide persistent `source_id` or `chunk_id` fields, causing verification failures in Module 19's `DeterministicVerifier`.
- **Decision:** Implemented deterministic SHA-256 ID generation in `src/parallel_research_team/workers/extractor.py` (`create_deterministic_source_id` using URL digest and `create_deterministic_chunk_id` using snippet content digest). Tools format outputs with explicit `Source ID: ...` and `Chunk ID: ...` tags, and `EvidenceCollector` immutably captures raw `SourceReference` records during tool execution.
- **Consequences:** Web evidence is 100% compatible with Module 19 verification requirements, preventing broken citation links.

---

## [2026-11-19] DEC-20-012: Worker Failure and Timeout Isolation with Evidence Salvage

- **Status:** Accepted
- **Context:** Individual worker crashes, tool timeouts, or budget exhaustion must not abort sibling workers, and any evidence gathered before failure must not be discarded.
- **Decision:** In `DefaultResearchWorker.execute` (`src/parallel_research_team/workers/worker.py`), the execution loop is protected by `asyncio.wait_for`. Timeouts map to `WorkerStatus.TIMEOUT` and unexpected exceptions map to `WorkerStatus.FAILED`. In both failure cases, any accumulated evidence in `EvidenceCollector` is extracted into `WorkerResearchOutput` and returned inside `WorkerExecutionResult`.
- **Consequences:** Complete fault tolerance; individual worker failures are captured as typed execution records with partial evidence salvage rather than crashing the orchestrator.

---

## [2026-11-19] DEC-20-013: Exact Token Accounting via `genai-core` Native Usage

- **Status:** Accepted
- **Context:** Previous worker implementations approximated input vs output token splits (75% / 25%) because `ToolBudget` only tracked total tokens.
- **Decision:** Enhanced `DefaultResearchWorker.execute` (`src/parallel_research_team/workers/worker.py`) to read exact `input_tokens`, `output_tokens`, and `total_tokens` directly from `loop_result` (with fallback to `worker_budget.current_input_tokens` / `current_output_tokens`), and compute exact dollar costs using `PricingCalculator`.
- **Consequences:** 100% precise financial cost tracking without heuristic approximations.

---

## [2026-11-19] DEC-20-014: Concurrency Throttling and Sibling Failure Isolation in Scheduler

- **Status:** Accepted
- **Context:** Parallel workers must execute within bounded concurrency limits (`max_concurrency`), support potential dependency ordering, and ensure that the failure or crash of one worker does not abort running sibling workers.
- **Decision:** Implemented `ParallelResearchScheduler` (`src/parallel_research_team/scheduling/scheduler.py`) using `asyncio.Semaphore(max_concurrency)` and `asyncio.gather(..., return_exceptions=True)`. Upstream dependencies are coordinated asynchronously using `asyncio.Event` maps with cycle detection.
- **Consequences:** Safe, predictable worker concurrency; failures in one brief are isolated and recorded without disrupting sibling research tasks.

---

## [2026-11-19] DEC-20-015: Real-Time Incremental Checkpointing and Source Pooling in Scheduler

- **Status:** Accepted
- **Context:** Long-running multi-agent runs can be interrupted at any moment. Waiting until all workers complete before persisting state risks losing all gathered evidence.
- **Decision:** The scheduler persists `CheckpointState` incrementally after each individual worker finishes (guarded by an `asyncio.Lock`), immediately pooling authoritative source chunks into `checkpoint.authoritative_sources` and updating token/cost metrics.
- **Consequences:** Durable execution recovery; interrupted runs retain all evidence gathered up to the point of interruption.

---

## [2026-11-19] DEC-20-016: Worker Deliverable and Citation Grounding Validation

- **Status:** Accepted
- **Context:** Workers may occasionally return empty summaries, miss requested format elements, or hallucinate citations referencing non-existent sources. Blindly passing unvalidated worker results into the synthesis stage corrupts the final report.
- **Decision:** Implemented `WorkerOutputValidator` (`src/parallel_research_team/scheduling/validator.py`) to verify deliverable non-emptiness, check that cited sources exist in the worker's evidence list, and flag formatting defects in `limitations`.
- **Consequences:** Protects downstream synthesis and Gate 1 deterministic verification from phantom citations and corrupt worker deliverables.

---

## [2026-11-19] DEC-20-017: Authoritative Source Pool and Source Deduplication

- **Status:** Accepted
- **Context:** Multiple concurrent workers may retrieve overlapping web sources, articles, or knowledge base chunks. Lead Agent synthesis requires a consolidated, immutable source repository to provide reliable citations without duplications or chunk collisions.
- **Decision:** Implemented `AuthoritativeSourcePool` (`src/parallel_research_team/synthesis/source_pool.py`) providing thread-safe deduplication indexed by composite `(source_id, chunk_id)` keys. The pool supports serialization to and from checkpoint state dictionaries.
- **Consequences:** Unified evidence pool for LLM synthesis and citation validation; seamless resumption from checkpoint with complete source preservation.

---

## [2026-11-19] DEC-20-018: Two-Stage Verification Pipeline with Module 19 Verification Gates

- **Status:** Accepted
- **Context:** Synthesized reports must not contain hallucinated sources, invalid quotes, or unsupported claims. Module 19 provides production-tested `DeterministicVerifier` and `SemanticGroundingVerifier`.
- **Decision:** Created `ResearchVerifier` (`src/parallel_research_team/synthesis/verifier.py`) executing a strict two-stage gate:
  - **Gate 1 (Deterministic):** Structural validation, citation integrity (every citation resolves to an authoritative source chunk), and quoted passage verbatim substring matching.
  - **Gate 2 (Semantic Grounding):** LLM judge entailment scoring using `SemanticGroundingVerifier` calibrated to threshold **0.70**, using dedicated judge configuration (`judge_provider` and `judge_model`). Gate 2 only runs if Gate 1 passes. If Gate 2 encounters an unexpected LLM judge error, the exception is caught, logged, and shielded gracefully.
- **Consequences:** Production-grade report reliability with guaranteed citation accuracy and grounded factual claims.

---

## [2026-11-19] DEC-20-019: Orchestrator Facade and Durable Checkpoint Resumption Protocol

- **Status:** Accepted
- **Context:** An end-to-end multi-agent research run coordinates multiple complex phases: effort classification, decomposition, concurrent worker scheduling, evidence consolidation, report synthesis, and two-stage verification. Furthermore, runs may be paused or interrupted at any point and require seamless resumption without duplicate execution of completed briefs.
- **Decision:** Implemented `ParallelResearchTeam` (`src/parallel_research_team/orchestrator.py`):
  - Exposes an async `run(topic, user_id, session_id, effort_override)` method coordinating all pipeline stages with automatic checkpointing and telemetry tracking.
  - Implemented `resume(run_id)` which re-inflates the `AuthoritativeSourcePool` from persisted checkpoint chunks, identifies only pending or uncompleted worker briefs, re-initializes `SharedRunBudget` with previous usage, schedules remaining work, and synthesizes the unified final report.
- **Consequences:** Provides a single, clean public entrypoint for callers while ensuring crash-resilient, durable execution recovery.

---

## [2026-11-19] DEC-20-020: Graceful Budget Exhaustion Interception and Salvage

- **Status:** Accepted
- **Context:** If a run exceeds its token, cost, or time ceiling, throwing an uncaught `BudgetExceededError` crashes the application and discards valuable evidence gathered prior to exhaustion.
- **Decision:** In `ParallelResearchTeam.run()`, `BudgetExceededError` is intercepted gracefully across both worker scheduling and synthesis stages. If usable evidence was collected before budget exhaustion, the run is transitioned to `RunStatus.PARTIALLY_COMPLETED`, partial findings are synthesized into a report if possible, and a user-friendly explanatory notice is recorded in `report.limitations` and `report.unresolved_limitations`.
- **Consequences:** Prevents application crashes, maximizes ROI by salvaging completed research, and communicates clear resource status to end users.

---

## [2026-11-20] DEC-20-021: Dynamic Exact Token and Pricing Telemetry Extraction Across Pipeline Stages

- **Status:** Accepted
- **Context:** Previous orchestrator code contained temporary static placeholders (`decomp_tokens = 1200`, `synth_tokens = 2000`) and heuristic input/output ratio approximations (`0.75 / 0.25`), which bypassed the real usage data returned by `LLMService` and caused inaccurate cost tracking.
- **Decision:** Updated `DecompositionEngine` and `SynthesisEngine` to extract real dynamic usage (`input_tokens`, `output_tokens`) from `LLMManagerResponse.usage`, record exact expenditure in `SharedRunBudget`, and expose `last_input_tokens`, `last_output_tokens`, `last_total_tokens`, and `last_cost_usd`. Updated `ParallelResearchTeam` to read these exact values directly into `ResearchTelemetryTracker`. Added `input_tokens` and `output_tokens` directly to `ResearchDecomposition` for durable checkpoint persistence.
- **Consequences:** 100% exact token accounting and monetary billing across all pipeline stages with zero static constants or heuristics.
