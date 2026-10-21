# GOAL.md — `genai-core`

> **Document type:** Purpose, intended outcomes, scope, and architectural direction  
> **Status:** Living design document  
> **Intended location in the repository:** `packages/genai-core/docs/GOAL.md`

## 1. Executive goal

Build `genai-core` as a small, reusable, well-tested Python package that provides the common GenAI capabilities needed across the `genai-engineer-journey` modules and future applications.

The package exists to turn repeated, project-specific implementations into shared, clearly defined capabilities. It should make it easier to build the next mini-project without copying service wrappers, recreating provider configuration, inventing another response format, or debugging inconsistent imports.

**`genai-core` is not just an LLM wrapper.** LLM integration is the first practical milestone. The broader goal is a dependable foundation that can gradually support model access, embeddings, retrieval, memory, tool/function-calling primitives, and other cross-project capabilities when real projects demonstrate the need for them.

The package must grow from proven reuse—not from a desire to implement every GenAI concept up front.

## 2. Why create it before Module 17?

Modules 0–16 have already produced learning and mini-project code. As the journey moves into Modules 17–20, projects are expected to connect more closely and reuse concepts built earlier. Without a shared package, each new project can introduce another copy of the same logic.

The problems `genai-core` is intended to prevent include:

- **Duplicated implementations:** Similar LLM, embedding, retrieval, or memory logic gets copied from one mini-project to another.
- **Inconsistent contracts:** Similar components accept different arguments, return different shapes, or report failures differently.
- **Import friction:** Code copied between folders develops path assumptions and becomes hard to import or test reliably.
- **Provider coupling:** Application logic becomes dependent on one provider's SDK or response structure.
- **Fixes do not propagate:** A correction made in one copy is not automatically available to other projects.
- **Unclear ownership:** It becomes difficult to decide whether a behavior belongs in a mini-project or a shared component.
- **Expensive future changes:** Switching providers, adding tests, or improving error handling requires updates across multiple projects.

Creating the shared package before Module 17 gives the connected modules a common starting point. It does **not** mean every capability must be fully designed or implemented before the learning projects begin.

## 3. Mission and intended outcomes

### 3.1 Primary outcomes

`genai-core` should:

1. **Reduce duplication.** Put genuinely shared, stable logic in one maintained package.
2. **Standardize contracts.** Give common capabilities predictable inputs, outputs, configuration, and error behavior.
3. **Separate application logic from provider details.** Keep provider-specific request/response handling behind adapters or equivalent boundaries where that separation is useful.
4. **Support composition.** Allow capabilities to be used independently or combined without requiring one monolithic framework.
5. **Improve testability.** Make core behavior testable without requiring live provider calls for every unit test.
6. **Make imports and packaging reliable.** Let mini-projects depend on the package through a supported public API instead of copying files or importing private internals.
7. **Enable incremental evolution.** Add functionality as Modules 17–20 and later work reveal repeatable needs.
8. **Preserve learning value.** The package should simplify repeated infrastructure without hiding the important concepts being studied in each module.

### 3.2 What success looks like

A developer starting a new mini-project should be able to reuse an existing `genai-core` capability, provide the required configuration, and focus on the project's unique behavior. They should not need to copy a previously implemented service or reverse-engineer a different interface from an earlier project.

When a shared behavior needs to change, the change should have one clear home, relevant tests, documented compatibility implications, and a predictable way to reach the projects that use it.

## 4. Who and what is it for?

The initial consumers are the connected mini-projects in Modules 17–20 of `genai-engineer-journey`. Future consumers may include other modules, demonstrations, experiments, and standalone applications that have a genuine need for the same capabilities.

The package should be useful for:

- shared model access and normalized generation behavior;
- reusable embedding interfaces and provider integrations;
- reusable retrieval abstractions and adapters;
- explicit memory primitives and storage/retrieval integrations;
- reusable tool/function-call representations and validation boundaries;
- shared configuration, errors, typed contracts, and provider integration conventions;
- other cross-project functionality that proves stable and reusable.

These are capability areas and an evolutionary direction—not a mandate to implement all of them in the first release.

## 5. Scope boundaries

### 5.1 What belongs in `genai-core`

A capability is a candidate for the package when it is used, or is very likely to be used, by multiple projects; has a meaningful and stable contract; can be explained independently of one project's domain; and can be tested and maintained as shared code.

Examples include provider adapters, reusable service logic, shared request/response models, common error types, configuration validation, and small utilities that are essential to those capabilities.

### 5.2 What should stay in an individual mini-project

Project-specific behavior should remain in the project that owns it. Examples include:

- a particular application's business workflow or user journey;
- a domain-specific prompt or policy that is not shared;
- application-specific orchestration and control flow;
- project-specific data models and persistence rules;
- experiments that have not yet demonstrated repeatable reuse;
- presentation, API routes, CLI commands, and deployment configuration for one application.

A mini-project may wrap or compose `genai-core` to express its own workflow. The shared package should not absorb that workflow simply because it uses an LLM.

### 5.3 What `genai-core` is not intended to become

`genai-core` is not intended to be:

- a replacement implementation of every feature in LangChain, LangGraph, LlamaIndex, or other established frameworks;
- a universal abstraction that pretends all providers have identical capabilities;
- a monolithic application or a global runtime that every project must use;
- a place for every helper that happens to be written in Python;
- a speculative framework for capabilities that no project currently needs;
- a reason to remove useful framework-specific learning from a module.

Existing libraries and frameworks can still be used by individual projects or through focused adapters when they provide clear value. The package should integrate with them deliberately rather than trying to replace them.

## 6. Development strategy: start small, expand with evidence

The implementation should proceed in validated increments. The sequence below is a default direction; actual needs and module requirements can adjust it.

### Current Implementation Status

| Milestone / Capability | Status | Verified Deliverables |
|---|---|---|
| **Phase 0 — Package Boundary** | **Completed** | Clean public API (`genai_core`), package structure, automated pre-push checks (`ruff`, `mypy`). |
| **Phase 1 — LLM Service Integration** | **Completed** | Unified `LLMService`, provider adapters (Google, OpenAI, Anthropic), structured responses via Pydantic, tool calling, streaming, error normalization, and live integration tests. |
| **Observability Infrastructure** | **Completed** | Reusable, caller-controlled logging (`ObservabilityConfig`), zero side effects by default (`NullHandler`), JSON/Text formatters, secret scrubbing (`RedactingFilter`), non-throwing async context (`observability_context`), and fault-isolated logging (`safe_log`). |
| **Phase 2 — Embedding Service & Shared Resilience** | **Completed** | Full embedding capability across Local (`sentence-transformers`), OpenAI (`text-embedding-3-small`), and Google (`text-embedding-005`), non-blocking thread execution, strict dimension validation (`EmbeddingDimensionError`), internal batch slicing with 1-to-1 order preservation, shared async resilience (`RateLimiter` dual token buckets for RPM/TPM, `RetryPolicy` with backoff/jitter/deadlines, `extract_retry_after`), honest token usage reporting, and full observability integration. |
| **Phase 3 — Retrieval Abstractions, Hybrid Search & Re-ranking** | **Completed** | Full retrieval subsystem: Core data models (`DocumentChunk`, `RetrievalQuery`, `RetrievalResult`, `RetrievalResponse`), typed exceptions, lifecycle observability events, `VectorStoreAdapter` protocol, zero-dependency `InMemoryVectorStore`, production-grade `PGVectorStoreAdapter` (parameterized `<=>` cosine distance, full-text search via `websearch_to_tsquery` and `ts_rank_cd`, JSONB containment filtering, batch upsert, delete), `RetrievalService` (dense, sparse, and concurrent hybrid search via score fusion), and optional neural re-ranking via `CrossEncoderReranker`. |
| **Phase 4 — Memory Subsystem & Automatic Knowledge Consolidation** | **Completed** | Unified short-term (turns, pinned goals, token budget pruning), long-term (semantic vector search, recency decay, supersession, purge), and episodic memory (canonical SHA-256 fingerprinting, idempotency checks), pluggable in-memory and PostgreSQL storage adapters, plus automatic knowledge consolidation (dual context input via `session_id` or `WorkingContext`, LLM-driven fact extraction, sensitive data scrubbing, multi-tier deduplication, and structured `ConsolidationResult`). |
| **Phase 5 — Tool System Architecture & Bounded Orchestration** | **Completed** | Complete tool subsystem: `BaseTool`, `@tool` decorator, `ToolRegistry` with risk and tag filtering, `ToolExecutor` (concurrency semaphore, per-tool timeout, output size truncation, process-level isolation notice, error normalization), `LoopDetector` (canonical hashing), `ApprovalStore` protocol, `InMemoryApprovalStore`, and bounded `ToolLoopRunner` (step caps, deadline, token/cost budgets, pause on approval required, resume on decision). |

---

### Phase 0 — Establish the package boundary

- Confirm the package location and how projects install or import it in the repository.
- Define a small public API and keep internal implementation details private.
- Establish configuration, error, testing, and dependency conventions.
- Review what Modules 17–20 actually need before designing abstractions around assumptions.

**Outcome:** Completed. A real, importable package boundary and a documented method for deciding what is shared.

### Phase 1 — LLM service integration

This is the starting capability because model access is a common dependency of many GenAI projects.

- Define a clear request/response contract for the required generation use cases.
- Isolate provider-specific calls behind an adapter or provider integration boundary.
- Validate configuration and inputs before issuing a provider request.
- Normalize common response information while preserving meaningful provider differences.
- Define useful error categories and safe handling of credentials.
- Add unit tests using fakes/mocks and a separately controlled integration-test path.
- Demonstrate use from a small consumer project.

**Outcome:** Completed. One reusable, testable model-access path implemented via `LLMService` and validated through 56 unit tests and live integration tests across Google, OpenAI, and Anthropic.

### Observability — Cross-cutting infrastructure

Built as a reusable, caller-controlled capability in `genai_core.observability`.
- Consuming applications explicitly configure destination files or console output.
- Automatic secret and credential masking protects API keys while preserving token metrics.
- Async request tracing via `observability_context` correlates operations without intrusive argument passing.
- Fault-tolerant dispatch via `safe_log` prevents logging errors from impacting application requests.

**Outcome:** Completed. Fully tested and integrated across all `LLMService` operations.

### Phase 2 — Embedding service and shared resilience

Provide a complete, production-grade vectorization capability with unified APIs, multi-provider support, and reusable resilience:

- Implemented `EmbeddingService` entry point, `EmbeddingRequest`, and `EmbeddingResponse`.
- Established `EmbeddingProvider` runtime checkable protocol.
- Implemented `LocalEmbeddingProvider` using `sentence-transformers` as an optional package extra (`genai-core[local]`), running inference non-blocking via `asyncio.to_thread`.
- Implemented `OpenAIEmbeddingProvider` using `AsyncOpenAI` SDK with model `text-embedding-3-small`, flexible dimension truncation, order preservation via API index sorting, and honest usage reporting.
- Implemented `GoogleEmbeddingProvider` using current `google-genai` SDK with `text-embedding-005` (768 native dims) and `output_dimensionality` configuration. Prohibited retired `text-embedding-004`.
- Built package-level shared resilience package (`genai_core.resilience`):
  - In-memory async rate limiter (`RateLimiter`, `AsyncTokenBucket`) with RPM and TPM token bucket controls and `Retry-After` header parsing.
  - Internal heuristic token estimation (`estimate_tokens`, ~4 chars/token) strictly for internal TPM pacing; never exposed as actual response usage.
  - Asynchronous retry policy (`RetryPolicy`, `execute_with_retry`) with exponential backoff, full jitter, attempt timeouts, overall bounded request deadlines, and selective retryability (fast-fail for auth/quota/dimension errors).
- Enforced strict dimension validation during service initialization, rejecting unsupported dimensions via `EmbeddingDimensionError`.
- Added internal sub-batch slicing with strict 1-to-1 order preservation.
- Enforced empty input validation (`input=[]` or whitespace-only raises `ValueError`).
- Integrated lifecycle observability logging (`EMBEDDING_REQUEST_START`, `EMBEDDING_REQUEST_SUCCESS`, `EMBEDDING_RATE_LIMITED`, `EMBEDDING_RETRY`, `EMBEDDING_TIMEOUT`, `EMBEDDING_REQUEST_FAILED`).

**Outcome:** Completed. Fully validated through 42 embedding and resilience unit tests (98 unit tests total in the package, 100% pass rate) and all pre-push checks (`ruff`, `mypy`).


### Phase 3 — Retrieval abstractions and integrations

Extract only the stable responsibilities needed across projects, separating reusable retrieval coordination from application-specific ingestion and database schemas:

- **Contracts, Protocols, Exceptions, Observability:**
  - Standardized retrieval data models: `DocumentChunk`, `RetrievalQuery`, `RetrievalResult`, `RetrievalResponse`.
  - Supported search modes: `search_mode in ("dense", "sparse", "hybrid")` with configurable weights (`vector_weight`, `keyword_weight`).
  - Established typed exceptions hierarchy: `RetrievalError`, `VectorStoreError`, `IncompatibleEmbeddingError`, `UnsupportedSearchModeError`, `RetrievalTimeoutError`.
  - Defined `@runtime_checkable` `VectorStoreAdapter` (`dimensions`, `search`, `search_keyword`, `upsert`, `delete`) and `Reranker` (`rerank`).
  - Structured observability events: `RETRIEVAL_START`, `RETRIEVAL_SUCCESS`, `RETRIEVAL_FAILED`, `RETRIEVAL_INDEX_START`, `RETRIEVAL_INDEX_SUCCESS`, `RETRIEVAL_DELETE_SUCCESS`, `RETRIEVAL_RERANK_START`, `RETRIEVAL_RERANK_SUCCESS`, `RETRIEVAL_RERANK_SKIPPED`.
- **In-Memory Store & Standalone Fusion:**
  - Zero-dependency `InMemoryVectorStore` using exact cosine similarity, term-frequency token keyword matching, dictionary subset metadata filtering, score thresholding, and upsert/delete operations.
  - Standalone mathematical fusion engine in `fusion.py`: min-max score normalization and weighted score blending.
- **PostgreSQL pgvector & Full-Text Search Adapter:**
  - Implemented `PGVectorStoreAdapter` conforming to `VectorStoreAdapter` using `psycopg` and `pgvector`.
  - Injected connection pattern: requires `connection`, `connection_pool`, or `connection_factory`, preventing ad-hoc connection churn.
  - Defensively validates all table and column names with whitelist regex (`^[a-zA-Z_][a-zA-Z0-9_]*$`).
  - Strict `%s` query parameterization on all inputs completely eliminates SQL injection risk.
  - Dense cosine similarity via `1 - (embedding <=> %s) AS similarity_score`.
  - Full-text keyword search via `ts_rank_cd` and `websearch_to_tsquery` supporting configured `search_vector_column` or dynamic fallback to `to_tsvector(%s, text_column)`.
  - Dictionary subset metadata filtering using PostgreSQL JSONB containment (`metadata @> %s::jsonb`) and optional dedicated tenant column support.
  - Batch upsert with `ON CONFLICT ({conflict_target}) DO UPDATE SET ...` and delete with `ANY(%s)`.
  - Flags transient operational errors with `retryable=True` in `VectorStoreError` for automatic retry.
  - Packaged as optional package extra `genai-core[postgres]`.
- **Retrieval Service & Optional Cross-Encoder Re-ranking:**
  - `RetrievalService` orchestrating text-to-vector embedding, direct vector querying, sparse keyword retrieval, concurrent hybrid search via `asyncio.gather`, weighted fusion, retry/timeout resilience, dimension validation, convenience `index_documents` auto-embedding, and safe observability dispatch.
  - Optional `CrossEncoderReranker` (`cross-encoder/ms-marco-MiniLM-L6-v2`) with worker thread execution via `asyncio.to_thread`.
  - Candidate pool expansion for high recall prior to re-ranking, score tracking in metadata, and graceful fallback handling (`warn_and_continue` vs `raise`).

**Outcome:** Completed. Fully validated through 154 unit tests (100% pass rate), a dedicated live integration test suite for embeddings and retrieval (validating real model inference, PostgreSQL/pgvector storage, keyword search, score fusion, and cross-encoder re-ranking), and all pre-push checks (`ruff`, `mypy`).

### Phase 4 — Memory primitives & Automatic Knowledge Consolidation

Implemented unified memory architecture supporting three distinct memory types with caller-selectable in-memory and PostgreSQL storage, plus automatic knowledge consolidation:
- **Short-term memory (`ShortTermMemory`)**: Conversation history, pinned goals that survive context pruning, dynamic token budgeting via `TokenCounter`, and session inactivity TTL.
- **Long-term memory (`LongTermMemory`)**: Persistent facts and user preferences, vector generation via `EmbeddingService`, hybrid semantic similarity and exponential recency decay search ($0.65 \times \text{sim} + 0.35 \times \text{recency} - 2.0 \times \text{is\_superseded}$), conflict resolution and supersession tracking (`supersedes_ids`), and temporal expiration.
- **Episodic memory (`EpisodicMemory`)**: Action execution history, canonical JSON parameter fingerprinting (SHA-256), idempotency decision checking (`execute` vs `do_not_repeat`), and rolling retention window purging.
- **Automatic Knowledge Consolidation (`consolidate`)**:
  - Dual context source resolution: callers supply either a `session_id` to retrieve history or an explicit `WorkingContext` slice (mutually exclusive).
  - LLM-driven durable fact extraction via `LLMMemoryExtractor` with reusable default prompt and custom prompt template support.
  - Sensitive information scrubbing (`validate_memory_content`) guarding against credential, token, SSN, and payment card leaks.
  - Multi-tier deduplication (exact normalized string matching and semantic cosine similarity thresholding) and supersession tracking.
  - Transparent `ConsolidationResult` reporting created, updated, skipped, and failed memory records.
- **Storage Adapters (`MemoryStorageAdapter`)**:
  - `InMemoryMemoryStore`: Zero-dependency, thread-safe in-memory store for unit testing and ephemeral agents.
  - `PostgresMemoryStore`: PostgreSQL + pgvector durable storage with CTE recency scoring, JSONB indexing, and strict multi-user isolation.
- **Unified Orchestration (`MemoryService`)**: High-level facade providing shortcut methods (`add_message`, `get_working_context`, `remember`, `recall`, `consolidate`, `record_action`, `check_action`, `purge_all`).

**Outcome:** Completed. Fully validated with 176 unit tests (100% pass rate), 4 dedicated live PostgreSQL integration tests verifying persistence, vector distance, recency scoring, and user isolation, and passing all pre-push checks (`ruff`, `mypy`).

### Phase 5 — Tool system architecture & bounded orchestration

Implemented a production-grade, framework-independent tool execution subsystem in `genai_core.tools`:
- **Definitions & Schemas**: `BaseTool` abstract base class and `@tool` decorator supporting both sync and async functions, automatic Pydantic schema generation from type hints, or explicit `args_schema`. Clean separation between declarative tool schema (`ToolSpec`) and execution logic (`run`).
- **Catalog & Discovery (`ToolRegistry`)**: Thread-safe registry (`RLock`) supporting dynamic tool registration, function wrapping, and query filtering by operational risk level, discovery tags, and exclusion lists.
- **Execution Engine (`ToolExecutor`)**: Isolated tool execution supporting sync functions in thread pools, concurrency limits via `asyncio.Semaphore`, per-tool execution timeouts via `asyncio.timeout`, output size truncation (`max_output_chars`) to prevent context window blowout, and complete error normalization (`VALIDATION_ERROR`, `EXECUTION_ERROR`, `TIMEOUT`). Includes explicit process-level boundary documentation without false claims of OS-level sandboxing.
- **Safety Policies & Guards (`genai_core.tools.guards`)**:
  - `LoopDetector`: Deterministic canonical SHA-256 argument hashing to prevent runaway or cyclical identical tool calls.
  - `ApprovalStore` protocol and reference `InMemoryApprovalStore` for persisting authorization state.
  - Risk categorization (`RiskLevel.READ_ONLY`, `REVERSIBLE_WRITE`, `IRREVERSIBLE`) with authorization checks gating high-risk operations.
- **Bounded Orchestration Loop (`ToolLoopRunner`)**: Multi-turn bounded tool execution coordinating `LLMService` and `ToolExecutor` with configurable step caps (`max_iterations`), timeouts, deadlines, and token budgets (`ToolBudget`), with full support for pausing on required approval and resuming with human-in-the-loop decisions.

**Outcome:** Completed. Fully hardened through deep cross-subsystem audit and verified with 213 unit tests and live integration tests across the package (100% pass rate) and passing all pre-push checks (`ruff`, `mypy`).

### Phase 6 — Further shared capabilities

Additional concerns such as caching, evaluation integration, or orchestration adapters may be considered when they solve a repeated problem. Each should have its own use case, contract, tests, and adoption justification.

**Important:** These phases do not mean all listed features are committed to the first release. Scope each phase against real Module 17–20 needs and keep unneeded capabilities out of the initial implementation.

## 7. Architectural principles

### 7.1 Reuse is earned

Do not extract code into the package only because it could theoretically be reused. Prefer a real repeated need, a clear contract, or a well-supported near-term requirement. Similar-looking code is not automatically the same abstraction.

### 7.2 Stable public contracts, flexible internals

Consumer projects should import supported public names. Internal module paths and provider SDK objects should not become accidental public contracts. Refactoring internals should not require unrelated projects to change unless a documented public contract changes.

### 7.3 Provider flexibility without pretending providers are identical

Keep common behavior consistent where possible. Expose provider-specific options or capabilities explicitly where necessary. Never silently discard a setting or pretend a provider supports a feature it does not.

### 7.4 Explicit configuration and dependencies

A consumer should be able to understand which provider, model, settings, and dependencies are being used. Avoid hidden global configuration and hidden network activity during import. Credentials should come from appropriate secret/configuration sources, not source code.

### 7.5 Framework-aware, not framework-dependent by default

Use a framework when it provides a real benefit, but do not force every shared primitive to depend on one orchestration framework. Core contracts should remain comprehensible and useful to ordinary Python consumers.

### 7.6 Test at the correct boundary

Unit tests should cover deterministic core behavior without live network calls. Adapter tests should verify provider translation. Integration tests may call real services only when explicitly configured and should be clearly separated from the default fast test suite.

### 7.7 Secure and predictable defaults

Do not log secrets or sensitive prompt content by default. Use bounded timeouts and explicit retry behavior. Make failures observable and understandable without exposing credentials or private data.

### 7.8 Backward compatibility matters

Because multiple modules may use the package at the same time, public changes should be intentional. Prefer additive changes when practical; document deprecations and breaking changes. Do not promise strict semantic stability for experimental APIs until they are explicitly promoted to supported status.

## 8. Dependency direction and ownership

The intended dependency direction is:

```text
Mini-project / application
        |
        v
 genai-core public API
        |
        v
 shared contracts and service logic
        |
        v
 provider / storage / framework adapters
        |
        v
 external SDKs and services
```

This is a conceptual direction, not a requirement that every capability use the same internal layering.

Rules:

- Consumer projects may depend on `genai-core`.
- `genai-core` must not import code from Module 17, Module 18, Module 19, Module 20, or another consuming application.
- A project should not reach into private core internals when a public API is available.
- Provider-specific SDKs should be confined to the integration boundary as far as practical.
- Avoid circular dependencies between core components; share small contracts in a neutral location when there is a real need.
- Optional integrations should not force every consumer to install all provider, vector-store, or framework dependencies.

## 9. How to decide whether a capability belongs in core

Before promoting a component into `genai-core`, answer these questions:

1. Do at least two projects use it, or is there a concrete near-term reason they will?
2. Is the behavior meaningfully the same across those projects?
3. Can we define an interface without encoding one application's workflow?
4. Can the component be tested independently and documented clearly?
5. Will centralization reduce maintenance rather than create excessive indirection?
6. Are its dependencies and provider-specific limitations understood?
7. Is the proposed abstraction needed now, or are we designing for hypothetical future use?

If the answers are uncertain, leave the code in the mini-project temporarily and revisit after another use case appears. Keeping a small amount of local duplication is sometimes cheaper than maintaining a premature abstraction.

## 10. Risks and intended responses

| Risk | Why it matters | Response |
|---|---|---|
| Overengineering | The package becomes a framework before the projects need one. | Implement the smallest capability that solves a real repeated problem. |
| Scope creep | Every GenAI-related helper gets added to core. | Apply the promotion checklist and keep application workflows local. |
| Provider leakage | Consumer code relies on one SDK's request/response shape. | Normalize shared behavior at the adapter boundary and expose differences explicitly. |
| Over-abstraction | The public API hides useful functionality or becomes hard to understand. | Prefer direct, typed contracts over excessive layers and generic factories. |
| Breaking changes | Connected modules fail when core evolves. | Test consumers, document public changes, and use deprecation/versioning discipline. |
| Dependency conflicts | Optional providers/frameworks burden unrelated projects. | Keep integrations modular and dependencies intentional. |
| False sense of portability | A common interface implies capabilities that providers do not share. | Document and validate feature support rather than silently emulating it. |
| Sensitive data leakage | Prompts, credentials, or memory data could appear in logs. | Redact by default and make any additional logging explicit. |
| Learning concepts get hidden | Core abstractions obscure the topic a module should teach. | Keep core focused on repeatable infrastructure; leave module-specific learning visible. |

## 11. Success criteria

`genai-core` is succeeding when the following become true:

- A new mini-project can use existing shared capabilities through documented imports instead of copying implementation files.
- The initial LLM integration has clear inputs, outputs, configuration, and errors, and can be tested without live API calls in its unit tests.
- Provider-specific code is separated from project workflow code to a useful degree.
- Shared changes can be tested against consumers and do not require manual edits to duplicate implementations.
- Modules 17–20 can compose the shared capabilities they genuinely need without being forced into one architecture.
- Core has no dependency on its consumers and avoids circular imports.
- Optional integrations do not impose unnecessary dependencies on all projects.
- New abstractions are justified by concrete needs and documented trade-offs.
- The package remains small enough that a contributor can understand where a capability belongs and how to extend it.

Success is **not** measured by the number of modules or abstractions inside the package. It is measured by reduced duplication, clearer contracts, safer change, and easier project development.

## 12. Relationship to `BUSINESS_LOGIC.md`

This document explains the **why, intended outcomes, scope, and direction** of `genai-core`. It does not by itself specify every runtime behavior.

`BUSINESS_LOGIC.md` defines the package's **responsibilities, behavioral rules, component boundaries, validation expectations, errors, and interactions**. Implementation details should follow those contracts where a capability is supported, while future capabilities remain planned until a real project justifies implementation.

## 13. Guiding statement

> Build a reusable GenAI foundation once, make its contracts understandable, and grow it only as connected projects prove what should be shared. Start with LLM service integration, but design `genai-core` for the broader set of reusable capabilities that future modules genuinely need.
