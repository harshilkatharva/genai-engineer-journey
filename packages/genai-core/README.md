# genai-core

Production-grade, modular, and resilient Python core package providing unified GenAI capabilities across `genai-engineer-journey` modules and downstream applications.

`genai-core` standardizes model access, embeddings, vector retrieval, rate limiting, retries, and privacy-safe observability behind clean, typed, and interchangeable public interfaces.

---

## Capabilities

### 1. Unified LLM Subsystem (`LLMService`)
- **Protocol-driven multi-provider abstraction**: Unified interface (`LLMProvider`) supporting:
  - `GoogleProvider`: Native Gemini models via `google-genai` SDK.
  - `OpenAIProvider`: OpenAI Chat & Structured Outputs via `openai` SDK.
  - `AnthropicProvider`: Claude Messages API via `anthropic` SDK.
- **First-class capabilities**:
  - `complete(prompt)`: Text completion with system instructions and temperature controls.
  - `complete_structure(prompt, response_schema)`: Schema-enforced structured output (Pydantic) with automated validation retries.
  - `complete_tool(messages, tools, tool_choice)`: Function declarations and multi-turn tool calling.
  - `stream(messages)`: Asynchronous streaming yielding normalized token deltas.
  - `complete_all(prompts)`: Concurrent, error-shielded batch completions.
- **Normalized responses**: `LLMResponseModel` unifying text, `structured_response`, `tool_calls`, token usage, model version, and latency.

### 2. Unified Embedding Subsystem (`EmbeddingService`)
- **Interchangeable providers**:
  - `LocalEmbeddingProvider`: Fast local embeddings via `sentence-transformers` (e.g. `all-MiniLM-L6-v2`, optional extra `[local]`).
  - `OpenAIEmbeddingProvider`: Hosted embeddings via `text-embedding-3-small` with configurable output dimensions and usage tracking.
  - `GoogleEmbeddingProvider`: Hosted embeddings via `text-embedding-005` with L2 normalization and `output_dimensionality` control.
- **Production reliability**:
  - Strict initialization dimension validation (rejects unsupported configurations fast).
  - Internal batch slicing with guaranteed 1-to-1 input sequence preservation.
  - Non-blocking async execution (`asyncio.to_thread` for local PyTorch inference).
  - Honest token usage accounting (never estimates or fabricates token usage).

### 3. Unified Retrieval Subsystem (`RetrievalService`)
- **Multi-modal retrieval workflows**:
  - `search_mode="dense"`: Semantic vector similarity search via embeddings and cosine distance.
  - `search_mode="sparse"`: Full-text keyword search via PostgreSQL `ts_rank_cd` and `websearch_to_tsquery` (or token matching in-memory), bypassing embeddings completely to save token costs.
  - `search_mode="hybrid"`: Concurrent execution of vector and keyword search via `asyncio.gather`, min-max score normalization, and weighted score fusion (`w_vector * s_vec + w_keyword * s_kw`).
- **Vector store adapters (`VectorStoreAdapter`)**:
  - `InMemoryVectorStore`: Zero-dependency, pure-Python vector store with exact cosine similarity, token keyword matching, metadata dictionary filtering, and score thresholding.
  - `PGVectorStoreAdapter`: PostgreSQL + pgvector adapter with dependency-injected connection pooling (`psycopg_pool`), full-text search (`ts_rank_cd` and `websearch_to_tsquery`), JSONB metadata containment filtering (`@>`), parameterized SQL, and identifier whitelisting.
- **Optional Cross-Encoder Re-ranking (`CrossEncoderReranker`)**:
  - Re-ranks candidate results jointly against the user query using `cross-encoder/ms-marco-MiniLM-L6-v2`.
  - Non-blocking execution via `asyncio.to_thread`.
  - Caller-controlled (disabled by default) with candidate pool expansion, score auditability in `metadata["retrieval_score"]`, and graceful fallback behavior.
- **Document Management**:
  - Chunk ingestion (`upsert`, `delete`, `index_documents`).
  - Automated vectorization of raw document chunks via attached `EmbeddingService`.
  - Strict dimension compatibility verification between embeddings and vector stores.

### 4. Unified Memory Subsystem (`MemoryService`)
- **Three distinct memory types**:
  - `ShortTermMemory`: Conversational history with turn tracking, pinned agent instructions, token budget snapshot pruning via `TokenCounter`, and session inactivity TTL.
  - `LongTermMemory`: Durable user profile facts and preferences with vector embedding, conflict resolution (`supersedes_ids`), temporal expiration, and hybrid cosine similarity + exponential recency decay ranking ($0.65 \times \text{sim} + 0.35 \times \text{recency} - 2.0 \times \text{is\_superseded}$).
  - `EpisodicMemory`: Historical action log with canonical JSON parameter fingerprinting (deterministic SHA-256), idempotency decision checking (`execute` vs `do_not_repeat`), and rolling retention window purging.
- **Automatic Knowledge Consolidation (`consolidate`)**:
  - Caller-controlled extraction of durable user facts, preferences, and goals from conversation history using `LLMService`.
  - Dual context source support: retrieve turns from `short_term` via `session_id`, or pass a caller-provided `WorkingContext` slice (mutually exclusive).
  - Customizable prompts via `custom_prompt_template`.
  - Sensitive information scrubbing (`validate_memory_content`) guarding against credential, token, SSN, and card leaks.
  - Multi-tier deduplication (exact matching and semantic cosine thresholding) and supersession tracking.
  - Returns structured `ConsolidationResult` reporting created, updated, skipped, and failed memory records.
- **Pluggable storage adapters (`MemoryStorageAdapter`)**:
  - `InMemoryMemoryStore`: Zero-dependency, thread-safe in-memory store for unit tests, ephemeral agents, and fast local runs.
  - `PostgresMemoryStore`: PostgreSQL + pgvector adapter with parameterized queries, pgvector distance operator (`<=>`), in-database CTE recency scoring, JSONB indexing, and strict multi-user isolation.
- **Unified orchestration facade**:
  - `MemoryService`: Coordinates short-term, long-term, and episodic memory with shortcut APIs (`add_message`, `get_working_context`, `remember`, `recall`, `consolidate`, `record_action`, `check_action`, `purge_all`).

### 5. Unified Tool Subsystem (`genai_core.tools`)
- **Declarative tool definitions**:
  - `BaseTool` abstract base class and `@tool` decorator supporting sync and async callables.
  - Automatic Pydantic schema synthesis from function type annotations or explicit `args_schema`.
  - Clean separation between schema validation (`ToolSpec`) and execution logic.
- **Thread-safe catalog & discovery (`ToolRegistry`)**:
  - Thread-safe tool catalog with lock protection (`RLock`).
  - Three-tier operational risk classification (`RiskLevel.READ_ONLY`, `RiskLevel.REVERSIBLE_WRITE`, `RiskLevel.IRREVERSIBLE`).
  - Dynamic schema generation filtered by tags, maximum risk levels, and exclusions.
- **Guarded execution engine (`ToolExecutor`)**:
  - Process-level schema validation, per-tool timeouts (`asyncio.timeout`), and concurrency limits (`asyncio.Semaphore`).
  - Output size protection: truncates outputs exceeding `max_output_chars` with clear warnings.
  - Fault normalization: unknown tools, validation failures, timeouts, and exceptions are normalized into structured `ToolResult` without crashing the application.
  - Transparent security boundaries: clearly documented process-level bounds without claiming OS-level sandboxing.
- **Infinite loop & thrashing protection (`LoopDetector`)**:
  - Deterministic SHA-256 parameter hashing over canonical key-sorted JSON representations.
  - Detects repeated identical invocations and injects diagnostic feedback into LLM conversation history.
- **Human-in-the-loop authorization & pause/resume**:
  - Gated authorization for high-risk operations (`require_approval_for={RiskLevel.IRREVERSIBLE}`).
  - Persistence protocol (`ApprovalStore`, `InMemoryApprovalStore`) storing pending requests and decisions.
  - Clean pause/resume: pauses loop when approval is required; resumes upon operator decision (`approved`/`rejected`) while preserving cumulative progress counters.
- **Bounded multi-turn orchestration loop (`ToolLoopRunner`)**:
  - Multi-resource budget enforcement: step caps (`max_iterations`), monotonic wall-clock deadlines, and token/cost budgets (`ToolBudget`).
  - Complete execution auditability in `ToolLoopResult`.

### 6. Shared Resilience Subsystem (`genai_core.resilience`)
- **Async Rate Limiter (`RateLimiter`)**:
  - Dual in-memory token buckets enforcing Requests Per Minute (RPM) and Tokens Per Minute (TPM).
  - Dynamic pause handling when upstream services return HTTP `Retry-After` headers.
- **Retry Policy (`RetryPolicy` & `execute_with_retry`)**:
  - Exponential backoff with randomized jitter and per-attempt timeouts.
  - Bounded overall deadlines across multi-batch and chained operations.
  - Intelligent classification: retries transient errors (HTTP 408/429/5xx, connection resets), fails fast on permanent errors (400, 401, 403, quota exhaustion, validation failures).

### 7. Caller-Controlled Observability (`genai_core.observability`)
- **Zero side-effects by default**: Configured with `logging.NullHandler()`; never creates log directories or files without explicit caller opt-in.
- **Automated redaction**: Sanitizes API keys (Google, OpenAI, Anthropic), Bearer headers, and sensitive fields before writing logs.
- **Context tracking**: Async-safe request correlation (`set_request_id`, `observability_context`).
- **Fault-tolerant dispatch**: `safe_log` prevents logging errors from interrupting primary GenAI workflows.

---

## Installation

### Base Installation
Installs core contracts, LLM service, in-memory vector store, and resilience:
```bash
uv pip install -e packages/genai-core
```

### Optional Extras
- **Local Embeddings** (includes `sentence-transformers` and PyTorch):
  ```bash
  uv pip install -e "packages/genai-core[local]"
  ```
- **PostgreSQL / pgvector Adapter** (includes `psycopg` and `pgvector`):
  ```bash
  uv pip install -e "packages/genai-core[postgres]"
  ```
- **All Extras**:
  ```bash
  uv pip install -e "packages/genai-core[local,postgres]"
  ```

---

## Configuration

Settings are managed via `Settings` and can be configured through `.env` or environment variables:

| Environment Variable | Description | Default |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | Google Gemini API key | `""` |
| `OPENAI_API_KEY` | OpenAI API key | `""` |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | `""` |
| `MAX_STRUCTURED_ERROR_RETRY` | Default retries on JSON schema validation errors | `2` |
| `MEMORY_WORKING_TTL_HOURS` | Inactivity TTL for short-term conversation sessions | `24` |
| `MEMORY_TOKEN_BUDGET` | Default token budget for working context snapshots | `4000` |
| `MEMORY_RECENCY_HALF_LIFE_DAYS` | Half-life in days for exponential recency decay | `30` |
| `MEMORY_EPISODIC_RETENTION_DAYS` | Rolling retention window in days for episodic actions | `14` |
| `INTEGRATION_TEST` | Enable live API integration test execution | `0` |
| `LOG_LEVEL` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `"INFO"` |
| `LOG_FILE` | Optional path to write rotating log files | `""` |

---

## Quickstart

### 1. LLM Service
```python
import asyncio
from pydantic import BaseModel
from genai_core import LLMService, GoogleProvider


class EntityExtraction(BaseModel):
    entities: list[str]
    sentiment: str


async def main():
    service = LLMService(default_provider=GoogleProvider(model="gemini-2.5-flash"))

    # Basic completion
    response = await service.complete("Explain vector embeddings in one sentence.")
    print(response.content)

    # Structured output
    structured = await service.complete_structure(
        prompt="Apple announced new MacBooks in Cupertino. Great feedback so far!",
        response_schema=EntityExtraction,
    )
    print(structured.structured_response)


asyncio.run(main())
```

### 2. Embedding Service
```python
import asyncio
from genai_core import EmbeddingService, LocalEmbeddingProvider


async def main():
    service = EmbeddingService(
        provider=LocalEmbeddingProvider(model="all-MiniLM-L6-v2"),
        dimensions=384,
        batch_size=32,
    )

    response = await service.embed(["Hello world", "Semantic search with genai-core"])
    print(
        f"Generated {len(response.embeddings)} vectors of dimension {len(response.embeddings[0])}"
    )


asyncio.run(main())
```

### 3. Retrieval Service (Dense, Sparse, Hybrid & Re-ranking)
```python
import asyncio
from genai_core import (
    CrossEncoderReranker,
    DocumentChunk,
    EmbeddingService,
    InMemoryVectorStore,
    LocalEmbeddingProvider,
    RetrievalQuery,
    RetrievalService,
)


async def main():
    embedding_service = EmbeddingService(
        provider=LocalEmbeddingProvider(),
        dimensions=384,
    )
    store = InMemoryVectorStore(dimensions=384)
    reranker = CrossEncoderReranker()
    retriever = RetrievalService(
        embedding_service=embedding_service,
        store=store,
        reranker=reranker,
    )

    # Index document chunks
    chunks = [
        DocumentChunk(
            id="c1",
            content="genai-core provides unified LLM, embedding, and retrieval access.",
        ),
        DocumentChunk(
            id="c2",
            content="PostgreSQL pgvector and full-text search enable hybrid similarity search.",
        ),
    ]
    await retriever.index_documents(chunks)

    # Hybrid search with cross-encoder re-ranking
    query = RetrievalQuery(
        query="hybrid search with pgvector",
        search_mode="hybrid",
        vector_weight=0.6,
        keyword_weight=0.4,
        rerank=True,
        top_k=2,
    )
    response = await retriever.search(query)
    for result in response.results:
        print(f"Rank {result.rank} | Score: {result.score:.4f} | Content: {result.content}")


asyncio.run(main())
```

### 3. Memory Service (Short-Term, Long-Term & Episodic)
```python
import asyncio
from genai_core import (
    EmbeddingService,
    InMemoryMemoryStore,
    LocalEmbeddingProvider,
    MemoryCategory,
    MemoryService,
    MessageRole,
)


async def main():
    # 1. Initialize store and embedding service
    store = InMemoryMemoryStore()
    embedding_service = EmbeddingService(provider=LocalEmbeddingProvider(), dimensions=384)
    memory = MemoryService(store=store, embedding_service=embedding_service)
    await memory.initialize()

    user_id = "user_42"
    session_id = "session_flight_booking"

    # --- Short-Term Working Memory ---
    await memory.pin_goal(user_id, session_id, "Target destination: Tokyo (HND/NRT)")
    await memory.add_message(user_id, session_id, MessageRole.USER, "Show me non-stop flights.")
    context = await memory.get_working_context(user_id, session_id, token_budget=1000)
    print(f"Working context messages: {len(context.messages)}, Pinned: {len(context.pinned_goals)}")

    # --- Long-Term Semantic Memory with Recency Decay ---
    await memory.remember(
        user_id,
        content="User holds Star Alliance Gold frequent flyer status.",
        category=MemoryCategory.FACT,
    )
    results = await memory.recall(user_id, query="frequent flyer benefits")
    print(f"Recalled: {results[0].record.content} (score={results[0].rank_score:.4f})")

    # --- Automatic Knowledge Consolidation via LLM ---
    # Caller can trigger consolidation by session_id or by passing a WorkingContext
    consolidation_result = await memory.consolidate(user_id=user_id, session_id=session_id)
    print(
        f"Consolidated: {len(consolidation_result.created)} created, "
        f"{len(consolidation_result.updated)} updated, "
        f"{len(consolidation_result.skipped)} skipped."
    )

    # --- Episodic Action History & Idempotency ---
    check = await memory.check_action(
        user_id, action_type="reserve_seat", parameters={"seat": "14A", "flight": "NH101"}
    )
    if check.status == "execute":
        await memory.record_action(
            user_id,
            action_type="reserve_seat",
            parameters={"seat": "14A", "flight": "NH101"},
            result_summary="Seat 14A confirmed",
        )
        print("Action executed successfully.")

    # Subsequent identical call is prevented automatically
    duplicate_check = await memory.check_action(
        user_id, action_type="reserve_seat", parameters={"seat": "14A", "flight": "NH101"}
    )
    print(f"Duplicate check recommendation: {duplicate_check.status} ({duplicate_check.reason})")

    await memory.close()


asyncio.run(main())
```

### 4. Tool System & Bounded Orchestration Loop

```python
import asyncio
from pydantic import BaseModel, Field
from genai_core import (
    ChatMessage,
    LLMService,
    RiskLevel,
    ToolBudget,
    ToolExecutor,
    ToolLoopRunner,
    ToolLoopStatus,
    ToolRegistry,
    tool,
)


class FlightQueryArgs(BaseModel):
    destination: str = Field(description="Destination airport or city code")
    max_price: float = Field(default=500.0, description="Maximum budget")


@tool(args_schema=FlightQueryArgs, risk_level=RiskLevel.READ_ONLY, tags=["travel"])
def search_flights(destination: str, max_price: float = 500.0) -> list[dict]:
    """Search available flights for destination under budget."""
    return [{"flight": "NH101", "to": destination, "price": 420.0}]


@tool(name="book_ticket", risk_level=RiskLevel.IRREVERSIBLE, tags=["booking"])
def book_ticket(flight_number: str) -> dict:
    """Book ticket - irreversible financial transaction requiring human approval."""
    return {"status": "confirmed", "booking_ref": "BOOK-992"}


async def main():
    registry = ToolRegistry([search_flights, book_ticket])
    executor = ToolExecutor(
        registry=registry,
        require_approval_for={RiskLevel.IRREVERSIBLE},
        default_timeout_s=10.0,
        max_output_chars=5000,
    )
    llm_service = LLMService()
    runner = ToolLoopRunner(llm_service=llm_service, executor=executor)

    # 1. Run loop with multi-resource budget (step caps, tokens, deadline)
    budget = ToolBudget(max_steps=5, max_tokens=2000)
    messages = [ChatMessage(role="user", content="Find flights to Tokyo and book NH101.")]

    result = await runner.run(messages=messages, budget=budget)

    # 2. Check if loop paused for approval on risky operation
    if result.status == ToolLoopStatus.APPROVAL_REQUIRED:
        print(f"Paused! Approval required for tool: {result.pending_approval.tool_name}")
        print(f"Arguments: {result.pending_approval.arguments}")

        # 3. Resume with authorized human decision
        from genai_core import ApprovalDecision

        decision = ApprovalDecision(approved=True, reason="User authorized booking via credit card")
        final_result = await runner.resume(
            decision=decision,
            request=result.pending_approval,
            messages=result.messages,
            budget=budget,
        )
        print(f"Final Outcome: {final_result.final_text}")


asyncio.run(main())
```

---

## Project Structure

```text
packages/genai-core/
├── docs/                     # Architectural single sources of truth
│   ├── GOAL.md               # Purpose, milestones, and progress
│   ├── FLOW.md               # End-to-end component and package flows
│   ├── DECISIONS.md          # Append-only architectural decisions
│   └── BUSINESS_LOGIC.md     # Confirmed contracts and validation rules
├── src/genai_core/
│   ├── core/                 # Settings and package configuration
│   ├── embedding/            # Local, OpenAI, and Google embedding providers
│   ├── exceptions/           # Strongly-typed exceptions
│   ├── llm/                  # LLM provider adapters and protocol
│   ├── memory/               # Short-term, long-term, and episodic memory + store adapters
│   ├── models/               # Pydantic data models (LLM, embedding, retrieval, memory, tools)
│   ├── observability/        # Logging, redaction, formatting, and tracing
│   ├── resilience/           # Rate limiting and retry policies
│   ├── retrieval/            # Vector store protocol, InMemory, PGVector, fusion, and reranker
│   ├── services/             # LLMService, EmbeddingService, RetrievalService, MemoryService
│   └── tools/                # Tool definitions, registry, executor, loop runner, and guards
├── tests/
│   ├── unit/                 # Fast, isolated unit tests (no live network)
│   │   ├── embedding/
│   │   ├── llm/
│   │   ├── memory/
│   │   ├── observability/
│   │   ├── retrieval/
│   │   └── tools/
│   └── integration/          # Live API and service integration tests (gated by INTEGRATION_TEST=1)
│       ├── embedding/
│       ├── llm/
│       ├── memory/
│       ├── observability/
│       ├── retrieval/
│       └── tools/
├── pyproject.toml
└── README.md
```

---

## Running Tests

### Unit Tests
```bash
uv run pytest packages/genai-core/tests/unit
```

### Live Integration Tests
Live tests require provider API keys and `INTEGRATION_TEST=1`:
```bash
INTEGRATION_TEST=1 uv run pytest packages/genai-core/tests/integration
```

### Pre-Push Quality Gate
Runs `ruff` rule checks, `ruff` formatting, and `mypy` strict type checking:
```bash
bash packages/genai-core/scripts/pre_push.sh
```
