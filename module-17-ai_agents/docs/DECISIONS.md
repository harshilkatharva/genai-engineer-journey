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

