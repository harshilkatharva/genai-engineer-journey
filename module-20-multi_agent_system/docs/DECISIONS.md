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
