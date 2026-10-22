# Module 17: Bounded Research and Support Agent — Architectural Decisions Log

## 2026-10-21: Initial Architecture and Foundation Decisions

### Decision 1: Single Reusable AI Infrastructure Dependency on `packages/genai-core`
- **Context:** Module 17 requires RAG retrieval, memory, tool orchestration, loop bounding, human approvals, and evaluation. Earlier modules (Module 9 RAG, Module 16 Memory) contained precursor implementations.
- **Decision:** Depend strictly and exclusively on `packages/genai-core`. Do not import or reuse any code from `module-09-rag` or `module-16-memory`.
- **Alternatives Considered:** Direct imports from Module 9 or 16. Rejected because `genai-core` was designed to consolidate and formalize these packages into unified, tested APIs.
- **Impact:** Eliminates redundant abstractions, guarantees unified type safety, and ensures changes to legacy modules do not break Module 17.

### Decision 2: Handling Missing Bulk / Specific Memory Deletions via Store Listing
- **Context:** In `genai-core`, `LongTermMemory` has single-record `delete(user_id, record_id)` but no `delete_all(user_id)`. `EpisodicMemory` has `delete_all(user_id)` but lacks single/specific episode deletion.
- **Decision:** As confirmed by the user, keep `packages/genai-core` strictly unmodified. Implement deletion orchestration within Module 17 tools:
  - `delete_longterm_memory_all`: Query `store.list_records(user_id)` and delete records individually via `long_term.delete(user_id, record.id)`.
  - `delete_episodic_memory_episodes`: Retrieve episodes via `get_episodes(user_id)`, remove target episodes, and write remaining back if needed or manage selective deletion in PostgreSQL.
- **Impact:** Adheres strictly to the change control rule forbidding unauthorized modifications to `packages/genai-core`.

### Decision 3: Lightweight Web Search Provider
- **Context:** Module 17 requires `search_web` without heavy search infrastructure or external API key requirements.
- **Decision:** Use DuckDuckGo HTML parsing via `httpx` and `BeautifulSoup4` with reasonable timeouts and character length limits, extracting core text and citations.
- **Alternatives Considered:** Tavily / SerpAPI / Google Custom Search. Rejected because no search API keys are configured in `.env`, and DDG provides zero-dependency public queries.
- **Impact:** Keeps the project self-contained and avoids external paid API dependencies for web search.

### Decision 4: PostgreSQL-Backed Approval Persistence (`PostgresApprovalStore`)
- **Context:** The agent requires pausing on irreversible operations, persisting the pending request, and resuming after an application restart. `genai-core` provides `InMemoryApprovalStore` which is volatile.
- **Decision:** Implement `PostgresApprovalStore` conforming to the `ApprovalStore` protocol from `genai-core`, storing approval requests and decisions in PostgreSQL. Also provide an in-memory fallback for isolated unit tests.
- **Impact:** Guarantees durable human-in-the-loop approvals across container and process restarts.

### Decision 5: Defense-in-Depth for Untrusted Tool Outputs
- **Context:** Content fetched from web pages or retrieved knowledge base chunks may contain adversarial prompt injections or delimiter escapes.
- **Decision:** Implement `ToolOutputSanitizer` combining:
  1. Semantic regex scanning for override patterns (`"ignore previous instructions"`, `"system:"`, etc.).
  2. Structural separation: wrapping untrusted content in XML delimiters `<untrusted_content source="...">...</untrusted_content>` with clear instructions to the LLM to treat it strictly as data.
- **Impact:** Prevents untrusted text from hijacking agent reasoning or bypassing approval boundaries.

### Decision 6: Mypy Overrides and Union Type Handling
- **Context:** `mypy` strict type checking encountered missing library stubs for local `genai-core` and BeautifulSoup `Tag.get()` returning `str | AttributeValueList | None`.
- **Decision:** Configure `pyproject.toml` with module overrides for `genai_core.*` and `bs4.*` with `ignore_missing_imports = true`, and explicitly narrow BeautifulSoup `href` values using runtime `isinstance(..., str)` checks.
- **Impact:** Ensures 100% clean type-checking with 0 errors across all 27 source and test files.

### Decision 7: Runnable Evaluation Benchmark Script (`scripts/run_eval.py`)
- **Context:** Requirement 9 specifies creating a runnable benchmark script reporting Success Rate, Mean Steps, and Mean Cost for the 15-goal evaluation suite, with deterministic mocks distinguishing offline testing from live-provider runs.
- **Decision:** Create `scripts/run_eval.py` executing `run_benchmark()` with `deterministic_mock_target`, printing a clean human-readable table with pass/fail breakdown per goal and aggregate KPIs.
- **Impact:** Enables instant reproducible evaluation without incurring API costs or requiring network access, verifiable in CI/CD and pre-push pipelines.

## 2026-10-22: Realistic PostgreSQL Integration and Gemini 3.5 Evaluation

### Decision 8: Database Seeding & Schema Architecture for Live Support Data
- **Context:** Live testing required actual relational tables in PostgreSQL for customer profiles (`support_customers`) and tickets (`support_tickets`), combined with vector-indexed documents (`document_chunks`) and persistent memory records (`memory_lt_records`, `memory_ep_episodes`).
- **Decision:** Implement `src/research_and_support_agent/data/db_store.py` for asynchronous CRUD operations using `psycopg`, and `src/research_and_support_agent/data/seed.py` for idempotent seeding with upsert statements (`ON CONFLICT`). Provide CLI script `scripts/seed_data.py`.
- **Alternatives Considered:** Raw SQL migrations vs. application-level table bootstrap. Chose `init_support_tables` with `CREATE TABLE IF NOT EXISTS` so tests and scripts self-initialize without external migration tools.
- **Impact:** Allows seamless database seeding and safe reruns without corrupting existing user records or failing on duplicate keys.

### Decision 9: Native Gemini 3.5 / 3.8 Support via `genai-core` Thought Signature Preservation
- **Context:** Google Gemini 3.5 and 3.8 reasoning models (`gemini-3.5-flash-lite`, `gemini-3.8-flash`) mandate cryptographic thought signatures (`Part.thought_signature`) generated during multi-step tool calls. Omitting these signatures in subsequent turns causes Google API 400 validation errors. Additionally, free-tier quotas enforce a 15 RPM ceiling.
- **Decision:** The dedicated AI agent updated `packages/genai-core`'s `ToolCall` and `GoogleProvider` to capture `thought_signature` on Turn 1, reconstruct it in subsequent `assistant` function call turns, enforce configurable request interval pacing (`min_call_interval`), and implement adaptive 429 retry backoff. Module 17 completely removed its temporary local wrapper (`GeminiLiveProvider`) and consumes `GoogleProvider` directly from `packages/genai-core`.
- **Alternatives Considered:** Retaining a local wrapper in Module 17. Rejected because reusable LLM infrastructure belongs exclusively in `genai-core`.
- **Impact:** Module 17 maintains zero custom LLM code, strictly adhering to the architectural boundary rule while fully supporting Gemini 3.5 and 3.8 models natively.

### Decision 10: Interactive Dual-Mode Evaluation Benchmark
- **Context:** The benchmark script needed to support both deterministic offline testing for CI/CD and live realistic evaluation against PostgreSQL, Gemini LLM, and DuckDuckGo web search. The user requested an interactive CLI prompt when running `run_eval.py` without flags.
- **Decision:** Updated `scripts/run_eval.py` to prompt the user with two explicit options:
  1. `1. deterministic_mock eval` (Offline, zero cost, deterministic mock target)
  2. `2. realistic eval run` (Live PostgreSQL database, Gemini 3.5 Flash Lite, DuckDuckGo search)
  Also preserved `--mock` and `--live` CLI flags for automated or non-interactive execution.
- **Impact:** Provides an intuitive developer experience while maintaining automated pipeline compatibility.

### Decision 11: Source Code Restructuring, Modular Test Architecture, and Conditional Integration Testing
- **Context:** The codebase needed better organization following conventions from `packages/genai-core` and `module-09-rag`. Flat test files (`tests/unit/test_domain_tools.py`, etc.) and top-level `models.py` / `config.py` needed modular separation into dedicated subdirectories. Additionally, integration tests needed to run conditionally based on `INTEGRATION_TEST=1` in `.env` or the environment, and real `EvalSample` flow tests were required.
- **Decision:**
  1. **Source Restructuring:**
     - Moved `models.py` to `src/research_and_support_agent/models/domain.py` with `models/__init__.py`.
     - Moved `config.py` to `src/research_and_support_agent/core/config.py` with `core/__init__.py`.
     - Updated all imports across `src/`, `scripts/`, and `tests/`.
  2. **Modular Test Architecture:**
     - Split unit tests into focused domain folders: `tests/unit/agent/`, `tests/unit/config/`, `tests/unit/data/`, `tests/unit/eval/`, `tests/unit/hitl/`, `tests/unit/memory/`, `tests/unit/models/`, `tests/unit/telemetry/`, and `tests/unit/tools/`.
     - Split integration tests into `tests/integration/agent/`, `tests/integration/eval/`, `tests/integration/hitl/`, and `tests/integration/memory/`.
     - Created `tests/integration/agent/test_eval_sample_flow.py` running real `EvalSample` instances (`goal_01_ticket_status`, `goal_02_customer_lookup`, `goal_03_kb_retrieval`) from the 15-goal benchmark.
  3. **Conditional Integration Hook:**
     - Configured `pytest_collection_modifyitems` in `tests/conftest.py` reading `.env` and environment for `INTEGRATION_TEST`.
     - When `INTEGRATION_TEST` is not in `("1", "true", "yes", "on")`, all tests under `tests/integration/` or marked `@pytest.mark.integration` are automatically skipped.
- **Alternatives Considered:** Using pytest custom CLI options (`--integration`). Rejected in favor of `.env` / environment variable `INTEGRATION_TEST=1` to allow standard zero-argument `pytest` or `pre_push.sh` executions to automatically run fast unit tests while permitting CI or explicit integration runs via `INTEGRATION_TEST=1 pytest`.
- **Impact:** Clean domain-driven directory layout, 0 type issues across 61 files, 58 fast unit tests running in ~0.5s, and 14 comprehensive integration tests running on-demand when `INTEGRATION_TEST=1` is provided (72 total tests).



