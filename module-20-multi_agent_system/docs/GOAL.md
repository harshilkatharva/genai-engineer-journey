# Module 20: Parallel Research Team — Goals & Implementation Plan

## 1. Project Goal

Build a production-grade, bounded **Parallel Research Team** multi-agent research system that decomposes broad, multi-faceted research questions into independent, typed worker briefs, executes them concurrently using bounded research workers (reusing Module 17's tool-calling loop execution engine), isolates worker failures to preserve partial findings, enforces shared run-level token and cost budgets, validates worker findings and citations using Module 19's verification architecture, stores execution state and research plans in `genai-core` memory and persistent checkpoints with durable resume capabilities, scales effort dynamically between direct single-agent execution and multi-worker fan-out, and delivers comprehensive end-to-end tracing and cost reporting.

The central design philosophy driving this module is:

> **Decompose broad questions into orthogonal, bounded research briefs, execute them in parallel with strict failure isolation and shared resource ceilings, validate and ground all evidence before synthesis, and preserve citations in the final verified report.**

---

## 2. Source of Truth

- `docs/GOAL.md` — Project milestones, completed items, and pending implementation tasks.
- `docs/BUSINESS_LOGIC.md` — Confirmed business rules, schemas, concurrency limits, budget envelopes, and verification thresholds.
- `docs/FLOW.md` — Runtime component interactions and execution traces (created and maintained as the application code is implemented).
- `docs/DECISIONS.md` — Append-only architectural decision log explaining why files, models, and flows were chosen (created and maintained during implementation).

---

## 3. Core Repository Integration & Boundary Analysis

### 3.1 Reused Architecture & Existing Packages

Module 20 builds upon established foundation libraries in the repository with clear boundaries:

1. **`packages/genai-core` Foundation:**
   - **`LLMService` ([llm_services.py](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/services/llm_services.py)):** Structured output generation (`complete_structure`) for typed decomposition plans and final report synthesis across Google, OpenAI, and Anthropic providers.
   - **`RetrievalService` ([retrieval_service.py](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/services/retrieval_service.py)):** Used directly by Module 20 (operating on its own corpus/chunks) with **dual-storage support**:
     - *Production / Live mode:* PostgreSQL via connection string using `PGVectorStoreAdapter`.
     - *Local / Test mode:* In-memory mode using `InMemoryVectorStore` for fast, reproducible, offline testing.
   - **`ToolLoopRunner` ([loop.py](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/tools/loop.py)) & `ToolExecutor` ([executor.py](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/tools/executor.py)):** Bounded multi-turn agent execution with iteration limits, loop detection, and tool execution.
   - **`ToolRegistry` ([registry.py](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/tools/registry.py)):** Cataloging and binding research tools (`search_web`, `search_knowledge_base`).
   - **`MemoryService` ([memory_service.py](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/services/memory_service.py)):** Unified memory manager managing `ShortTermMemory` (pinned research goals, session working context) and `LongTermMemory` (user domain preferences).
   - **Observability ([logger.py](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/observability/logger.py), [events.py](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/observability/events.py), [context.py](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/observability/context.py)):** Safe structured logging (`safe_log`), request ID context propagation, and PII redaction.
   - **Resilience ([rate_limiter.py](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/resilience/rate_limiter.py), [retry_policy.py](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/resilience/retry_policy.py)):** In-process token bucket rate limiting and exponential backoff retry policies.

2. **`module-17-ai_agents` Execution & Tool Capabilities:**
   - **`search_web` ([read_only.py](file:///home/harshilk/genai-engineer-journey/module-17-ai_agents/src/research_and_support_agent/tools/read_only.py)):** DuckDuckGo web scraping with HTML snippet extraction, URL cleaning, and prompt injection sanitization.
   - **`sanitize_tool_output` ([sanitizer.py](file:///home/harshilk/genai-engineer-journey/module-17-ai_agents/src/research_and_support_agent/tools/sanitizer.py)):** Safeguarding external tool inputs and untrusted web contents.
   - *Architectural Decision on Telemetry:* Module 20 implements its own `PricingCalculator` (`src/parallel_research_team/telemetry/pricing.py`) rather than depending on Module 17, providing native control over model pricing cards and rate updates.

3. **`module-18-planning` Abstractions & Patterns:**
   - **Planning & Verification Patterns:** Structured plan generation and validation techniques.
   - **Persistence Architecture:** Storage protocol pattern (`RunRecordStore` / `FileRunRecordStore`), adapted into real-time `CheckpointStore`.
   - *Architectural Limitation on Scheduling:* Module 18's `DAGScheduler` cancels all sibling tasks upon any failure. Module 20 implements a dedicated `ParallelResearchScheduler` providing complete failure isolation and partial result salvage.

4. **`module-19-reflection` Citation & Report Architecture:**
   - **Verification Gates ([deterministic_verifier.py](file:///home/harshilk/genai-engineer-journey/module-19-reflection/src/self_improving_report_writer/verification/deterministic_verifier.py), [semantic_grounding_verifier.py](file:///home/harshilk/genai-engineer-journey/module-19-reflection/src/self_improving_report_writer/verification/semantic_grounding_verifier.py)):**
     - Gate 1: Deterministic structural, citation pointer, and quoted passage substring verification.
     - Gate 2: LLM claim-passage entailment verification calibrated with a pass threshold of **0.70**.
   - *Architectural Decision on Report Model:* Module 20 defines its own `Report` model (`src/parallel_research_team/models/report.py`) with Module 20-specific fields (`research_team_version`, `brief_coverages`, `unresolved_limitations`) while preserving strict structural compatibility with Module 19's verifiers.

5. **Versioned Prompt Architecture:**
   - Prompt templates are maintained in dedicated versioned directories (`src/parallel_research_team/prompts/v1/`) with `metadata.json` and Jinja2 templates (`decomposition/`, `worker/`, `synthesis/`), managed by `PromptTemplateManager`.

---

## 4. Scope & Milestones

### Milestone 1: Domain Models, Shared Budget, Configuration & Checkpoint Storage
- [x] Define core Pydantic domain models in `src/parallel_research_team/models/`:
  - `WorkerBrief`, `BriefType`, `OutputFormatSpec`, `WorkerStatus`, `WorkerExecutionResult`, `WorkerResearchOutput`.
  - `ResearchDecomposition`, `EffortLevel`.
  - Module 20's own `Report`, `ReportSection`, `Claim`, `Citation`, `SourceReference` in `models/report.py`.
  - `CheckpointState`, `ResearchTeamRunResult`, `ExecutionMetrics`.
- [x] Implement centralized `ResearchTeamConfig` in `core/config.py`:
  - Centralizes all tunable runtime knobs: judge provider (`google`), judge model (`gemini-3.1-flash-lite`), default models (`gemini-3.5-flash-lite`), semantic pass threshold (`0.70`), `max_concurrency` (3), `max_workers` (4), `per_worker_timeout_seconds` (45.0s), `per_worker_max_steps` (5), `max_total_tokens` (60,000), `max_total_cost_usd` ($0.50), `prompt_version` ("v1"), and retrieval settings (`retrieval_mode`, `postgres_connection_string`).
- [x] Implement `SharedRunBudget` in `core/budget.py`:
  - Thread- and async-safe global budget tracking `current_tokens`, `current_cost_usd`, `start_monotonic`, `deadline_monotonic`.
  - Concurrency-safe atomic check (`check_budget`) and resource recording (`record_usage`) protected by `asyncio.Lock`.
- [x] Implement `CheckpointStore` protocol and `FileCheckpointStore` in `storage/`:
  - Atomic JSON persistence of `CheckpointState` under `checkpoints/checkpoint_<run_id>.json`.
  - Immediate state flush upon milestone events (decomposition complete, worker completion, synthesis complete).
- [x] Implement Module 20 `PricingCalculator` in `telemetry/pricing.py`.
- [x] Implement test structure matching Module 18/19:
  - `tests/conftest.py` with integration gating, sample sources, and test configuration.
  - Unit tests covering domain models, budget race conditions under concurrency, and checkpoint serialization/deserialization.

### Milestone 2: Effort Scaling, Prompt Templates & Lead Agent Decomposition
- [x] Implement `PromptTemplateManager` in `prompts/manager.py` and create `prompts/v1/`:
  - `metadata.json`, `decomposition/system.jinja2`, `decomposition/user.jinja2`, `worker/system.jinja2`, `worker/user.jinja2`, `synthesis/system.jinja2`, `synthesis/user.jinja2`.
- [x] Implement `DecompositionEngine` in `decomposition/engine.py`:
  - Structured prompt using `LLMService.complete_structure` with Pydantic `ResearchDecomposition` schema.
  - Effort classification: `DIRECT` (simple, narrow, or factual queries requiring 1 worker) vs `PARALLEL_DECOMPOSED` (multi-faceted, comparative queries requiring 2–5 workers).
  - Explicit boundary definition: generating orthogonal sub-topics with mutually exclusive scopes, search guidelines, and typed `OutputFormatSpec`.
- [x] Implement `DecompositionValidator` in `decomposition/validator.py`:
  - Verify worker count bounds ($1 \le N \le \text{max\_workers}$).
  - Verify non-empty objectives and disjoint boundary specifications.
- [x] Memory integration:
  - Store research goal and decomposition plan as `PinnedGoal` in `genai_core.ShortTermMemory`.
  - Retrieve long-term user preferences from `genai_core.LongTermMemory`.
- [x] Unit and integration tests covering decomposition of simple vs broad research questions, boundary enforcement, and memory saving.

### Milestone 3: Bounded Research Worker, Extensible Registry & Tool Suite
- [x] Implement extensible `ResearchWorkerProtocol` and `WorkerRegistry` in `workers/registry.py`:
  - Interface-based factory pattern allowing registration of new worker types without modifying scheduler code.
- [x] Implement `ResearchWorker` in `workers/worker.py`:
  - Specialized research agent wrapping `genai_core.tools.loop.ToolLoopRunner`.
  - Bound to research tool registry (`search_web`, `search_knowledge_base`).
  - Strict per-worker execution envelope: `max_steps` (default: 5), `timeout_seconds` (default: 45s).
  - Pre-step and post-step integration with `SharedRunBudget`.
- [x] Implement `RetrievalService` factory with dual-storage support:
  - Configurable between PostgreSQL (`PGVectorStoreAdapter`) and in-memory store (`InMemoryVectorStore`).
- [x] Implement evidence ingestion & extraction in `workers/extractor.py`:
  - Transform `search_web` outputs (and KB chunks) into immutable `SourceReference` records with deterministic IDs (`url` hash + index/content hash).
  - Extract structured `WorkerResearchOutput` containing findings, claims, and cited source IDs.
- [x] Unit and integration tests for worker execution, tool calling, per-worker timeout enforcement, and source extraction.

### Milestone 4: Parallel Scheduling, Concurrency Control & Failure Isolation
- [x] Implement `ParallelResearchScheduler` in `scheduling/scheduler.py`:
  - Bounded concurrency execution via `asyncio.Semaphore(max_concurrency)`.
  - Async task dispatch using `asyncio.gather(..., return_exceptions=True)`.
  - **Strict Failure Isolation:** Worker failures (exceptions, timeouts, budget stops) do NOT cancel sibling workers.
  - Real-time checkpoint updates as individual workers complete.
  - Partial result recovery: salvage completed tool findings and sources from failed or timed-out workers.
- [x] Implement `WorkerOutputValidator` in `scheduling/validator.py`:
  - Validate worker deliverable structure and verify that cited sources exist in the worker's retrieved evidence pool.
- [x] Unit tests for worker isolation (one worker fails while others succeed), timeout handling, concurrency throttling, and checkpoint persistence during execution.

### Milestone 5: Lead Agent Synthesis & Citation Verification Gate
- [x] Implement `AuthoritativeSourcePool` in `synthesis/source_pool.py`:
  - Consolidate and deduplicate all validated `SourceReference` instances from successful and partial workers into a unified pool keyed by `(source_id, chunk_id)`.
- [x] Implement `SynthesisEngine` in `synthesis/engine.py`:
  - Synthesize collective research findings into Module 20's `Report` model.
  - Explicit reporting of missing or failed research: Document incomplete briefs in `Report.limitations` and `Report.unresolved_limitations`.
  - Strict preservation of source citations (`Citation` objects pointing to `source_id` and `chunk_id`).
- [x] Verification Integration in `synthesis/verifier.py`:
  - Gate 1: Run Module 19 `DeterministicVerifier` to ensure zero broken citations, valid quoted passages, and structural compliance.
  - Gate 2: Run Module 19 `SemanticGroundingVerifier` with threshold **0.70** to confirm claim-passage entailment.
- [x] Unit and integration tests for multi-worker evidence synthesis, citation verification, and limitation reporting.

### Milestone 6: Orchestrator Facade, Checkpointing & Resumption
- [x] Implement `ParallelResearchTeam` facade in `orchestrator.py`:
  - End-to-end orchestration coordinating decomposition, scheduling, validation, synthesis, and memory persistence.
  - Resumption protocol: `resume(run_id)` loads checkpoint, identifies uncompleted briefs, preserves completed worker outputs, and resumes execution seamlessly.
  - Budget error interception: catches `BudgetExceededError`, marks status `PARTIALLY_COMPLETED`, and appends friendly explanation to `limitations`.
- [x] Implement `ResearchTelemetryTracker` in `telemetry/tracker.py`:
  - Track per-worker and lead-agent tokens, execution duration, tool invocations, and USD cost via Module 20's `PricingCalculator`.
  - Produce comprehensive `ResearchTeamRunResult` with detailed execution summary and `ExecutionMetrics`.
- [x] End-to-end integration and unit tests verifying full lifecycle, effort scaling (direct vs parallel), interruption/resume fidelity, and cost tracking:
  - `tests/integration/test_integration_llm_service.py`: Live Gemini Developer API structured generation for `ResearchDecomposition` and `Report`.
  - `tests/integration/test_integration_retrieval.py`: In-memory similarity retrieval over real handbook chunks and PostgreSQL pgvector connectivity over 7,852 document chunks.
  - `tests/integration/test_integration_worker_and_verifier.py`: Live bounded worker execution and Gate 1 / Gate 2 reflection verification with `gemini-3.1-flash-lite` judge.
  - `tests/integration/test_integration_orchestrator.py`: Live end-to-end execution comparing Single-Agent `DIRECT` vs Multi-Agent `PARALLEL_DECOMPOSED` modes, exact token/cost tracking, and file checkpoint persistence.

---

## 5. Explicitly Deferred

- **Evaluation Pipeline:** Evaluation benchmarks, comparative datasets, and automated metric scoring are explicitly deferred until the functional system is completely implemented and verified.
