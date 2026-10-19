# Execution & Data Flow: `genai-core` Scope

This document details the actual verified execution flows, component interactions, and data normalization across the entire `genai-core` package (`packages/genai-core/`).

It is structured by service and capability, providing an evolving architecture map of how components connect.

---

## 1. Overall Package Architecture & Component Interconnection

`genai-core` provides reusable, provider-agnostic GenAI capabilities designed with clean layer separation, loose coupling, shared resilience, and cross-cutting observability:

```mermaid
flowchart TD
    Client[Consuming Application / Mini-Projects]

    subgraph PublicAPI ["Public API: genai_core"]
        direction TB
        LLM[LLMService]
        EMB[EmbeddingService]
        RET[RetrievalService]
        RESIL[Resilience: RateLimiter & RetryPolicy]
        OBS[Observability Engine]
        MODELS[Models: Request & Response Contracts]
        EXC[Exceptions: Typed Error Hierarchy]
        CORE[Core: Settings & Configuration]
    end

    Client -->|generate text / tool calls| LLM
    Client -->|vectorize text / documents| EMB
    Client -->|semantic search & index| RET
    Client -->|configure logging / request_id context| OBS
    Client -->|access config / typed models| MODELS

    subgraph ServiceLayer ["Service Orchestration Layer"]
        LLM --> LLMProviderResolve{Resolve LLM Provider}
        EMB --> EMBProviderResolve{Resolve Embedding Provider}
        RET --> RETStoreResolve{VectorStoreAdapter}
        RET --> RETEmbedQuery["EmbeddingService.embed_query()"]
    end

    subgraph ResilienceSubsystem ["Shared Resilience Infrastructure"]
        EMB --> RateLimiter[RateLimiter: RPM & TPM Token Bucket]
        EMB --> RetryPolicy[RetryPolicy: Exponential Backoff & Jitter]
        RET --> RetryPolicy
        RateLimiter -.->|reusable by| LLM
    end

    subgraph CrossCutting ["Cross-Cutting Observability & Safety"]
        OBS --> ContextManager[observability_context / request_id]
        OBS --> SafeLog[safe_log Fault Isolation]
        OBS --> Redaction[RedactingFilter Secret Scrubbing]
    end

    LLM -.->|emit lifecycle events| SafeLog
    EMB -.->|emit lifecycle events| SafeLog
    RET -.->|emit lifecycle events| SafeLog
    ResilienceSubsystem -.->|emit throttled / retry / timeout events| SafeLog

    subgraph Adapters ["Integration & Storage Adapters"]
        LLMProviderResolve --> GoogleLLM[GoogleProvider]
        LLMProviderResolve --> OpenAILLM[OpenAIProvider]
        LLMProviderResolve --> AnthropicLLM[AnthropicProvider]

        EMBProviderResolve --> LocalEmb[LocalEmbeddingProvider]
        EMBProviderResolve --> OpenAIEmb[OpenAIEmbeddingProvider]
        EMBProviderResolve --> GoogleEmb[GoogleEmbeddingProvider]

        RETStoreResolve --> InMemoryStore[InMemoryVectorStore: Zero-dependency]
        RETStoreResolve -.->|Stage 3| PGVectorStore[PGVectorStoreAdapter: PostgreSQL]
    end

    subgraph FutureServices ["Future Services (Planned)"]
        MEM[MemoryService]
        TOOL[ToolOrchestrator]
    end

    SafeLog --> Formatter[JsonFormatter / TextFormatter]
    Formatter --> Output[Caller-Defined File / Console]
```

### Component Roles & Responsibilities:
- **`genai_core.services`**: Unified entry points (`LLMService`, `EmbeddingService`, `RetrievalService`). Validate input, resolve providers/backends, coordinate internal batching, manage fault isolation, and emit observability events.
- **`genai_core.retrieval`**: Vector store abstractions and reference implementations:
  - `VectorStoreAdapter`: Runtime-checkable protocol defining `dimensions`, `search`, `upsert`, and `delete`.
  - `InMemoryVectorStore`: Zero-dependency in-memory reference store using exact cosine similarity, supporting metadata filtering and score thresholding.
  - `PGVectorStoreAdapter` *(Planned Stage 3)*: PostgreSQL pgvector adapter using connection injection.
- **`genai_core.resilience`**: Shared asynchronous resilience infrastructure (`RateLimiter`, `AsyncTokenBucket`, `RetryPolicy`, `execute_with_retry`, `extract_retry_after`, `estimate_tokens`). Implements in-memory token bucket pacing for RPM/TPM and bounded retries with jitter and deadlines. Reused across embeddings, retrieval, and LLMs.
- **`genai_core.llm`**: Adapters conforming to `LLMProvider` protocol translating between normalized contracts and vendor SDKs (Google, OpenAI, Anthropic).
- **`genai_core.embedding`**: Adapters conforming to `EmbeddingProvider` protocol (`LocalEmbeddingProvider`, `OpenAIEmbeddingProvider`, `GoogleEmbeddingProvider`).
- **`genai_core.observability`**: Reusable logging with caller-isolated outputs, secret/key scrubbing, asynchronous request context propagation, and zero side effects by default.
- **`genai_core.models`**: Pydantic v2 schemas (`ChatMessage`, `LLMManagerRequest`, `LLMResponseModel`, `EmbeddingRequest`, `EmbeddingResponse`, `DocumentChunk`, `RetrievalQuery`, `RetrievalResult`, `RetrievalResponse`, `ToolSpec`).
- **`genai_core.exceptions`**: Typed error hierarchies (`LLMError`, `EmbeddingError`, `EmbeddingDimensionError`, `EmbeddingTimeoutError`, `RetrievalError`, `VectorStoreError`, `IncompatibleEmbeddingError`, `UnsupportedSearchModeError`, `RetrievalTimeoutError`).
- **`genai_core.core`**: Environment settings loaded via `pydantic-settings` with explicit precedence and fallback defaults.

---

## 2. LLMService Execution Flow

The diagram below illustrates how client calls enter `LLMService`, get dispatched to provider adapters conforming to `LLMProvider`, and return normalized responses:

```mermaid
flowchart TD
    Client[Client / Caller] -->|LLMManagerRequest or helper methods| LLMService[LLMService]

    subgraph Service Layer ["packages/genai-core/src/genai_core/services/llm_services.py"]
        LLMService --> ProviderResolve{Resolve Provider}
        ProviderResolve -->|Unsupported| ErrorResp[Return LLMManagerResponse with LLMError]
        ProviderResolve -->|Resolved| RouteType{Schema or Tool/Message?}
    end

    subgraph Provider Adapters ["packages/genai-core/src/genai_core/llm/"]
        RouteType -->|response_schema is set| CompleteStructure[complete_structure]
        RouteType -->|tools or messages| CompleteTool[complete_tool]
        CompleteStructure --> ProviderImpl{Provider Adapter}
        CompleteTool --> ProviderImpl

        ProviderImpl -->|Google| GoogleProvider[GoogleProvider]
        ProviderImpl -->|OpenAI| OpenAIProvider[OpenAIProvider]
        ProviderImpl -->|Anthropic| AnthropicProvider[AnthropicProvider]
    end

    subgraph Response Normalization
        GoogleProvider -->|SDK Response| Norm[Build LLMResponseModel]
        OpenAIProvider -->|SDK Response| Norm
        AnthropicProvider -->|SDK Response| Norm
        Norm --> ServiceUnwrap[Wrap in LLMManagerResponse]
    end

    ServiceUnwrap --> Client
    ErrorResp --> Client
```

### 2.1 Request Dispatch & Routing
- **File**: [`packages/genai-core/src/genai_core/services/llm_services.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/services/llm_services.py)
- **Function**: [`LLMService.complete`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/services/llm_services.py#L55-L121)
- **Execution Flow**:
  1. Receives an [`LLMManagerRequest`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/models/llm_model/llm_manager_request.py).
  2. Resolves provider name via precedence: `request.provider` -> `self.default_provider` -> `settings.default_llm_provider` -> fallback `"google"`.
  3. Checks `self.providers.get(provider_name)`.
  4. If provider is `None`: immediately returns `LLMManagerResponse(error=LLMError(provider=provider_name, code="unsupported_provider", message=...))`.
  5. **Operation Routing**:
     - **If `request.response_schema is not None`**: extracts prompt and calls `await provider.complete_structure(prompt=prompt, response_schema=request.response_schema)`.
     - **If `request.response_schema is None`**: builds messages and calls `await provider.complete_tool(messages=messages, tools=request.tools, tool_choice=request.tool_choice)`.

### 2.2 Provider Adapters
- **Google Gemini (`GoogleProvider`)**: Uses `google-genai` SDK. Supports native JSON schema constraints, function calling declarations, and delta streaming.
- **OpenAI (`OpenAIProvider`)**: Uses `openai` SDK v1+. Supports strict structured schema responses (`text={"format": ...}`), tool schemas via `FunctionToolParam`, and event stream deltas.
- **Anthropic Claude (`AnthropicProvider`)**: Uses `anthropic` SDK. Extracts text blocks for JSON schema validation, maps tools into Anthropic tool specs, and streams text deltas.

### 2.3 Concurrent Provider Query Flow (`complete_all`)
- Takes `prompt` and optional `response_schema`.
- Spawns concurrent asynchronous tasks for each registered provider using `asyncio.gather(*tasks, return_exceptions=True)`.
- Returns a list containing `LLMResponseModel` instances or exceptions for failed providers without aborting successful ones.

### 2.4 Error Normalization Flow
- Provider exceptions are intercepted and inspected for HTTP status codes, error codes, and messages.
- Status codes in `{408, 429, 500, 502, 503, 504}` are flagged `retryable=True`.
- Returns `LLMManagerResponse(error=LLMError(...))` guaranteeing no unhandled exceptions leak to callers.

---

## 3. Observability Execution Flow

Observability is a cross-cutting subsystem in `packages/genai-core/src/genai_core/observability/`:

```mermaid
flowchart TD
    Caller["Consuming Application"]
    ConfigureCall["configure_observability(config)"]
    ContextCall["with observability_context(request_id)"]
    ObsEngine["Observability Subsystem"]

    Caller -->|1. Setup config| ConfigureCall
    ConfigureCall --> ObsEngine
    ObsEngine --> InitHandlers["Attach RotatingFileHandler / StreamHandler"]
    InitHandlers --> AttachRedactor["Attach RedactingFilter"]

    Caller -->|2. Scoped operation| ContextCall
    ContextCall --> SetVar["contextvars.set(request_id)"]

    subgraph ServiceLogging ["Service Invocation (LLM, Embedding, or Retrieval)"]
        Service["LLMService / EmbeddingService / RetrievalService"]
        Service --> SafeLog["safe_log(logger, level, event, **fields)"]
        SafeLog --> ExtractCtx["Read request_id from contextvars"]
        ExtractCtx --> EmitRecord["logger.log()"]
    end

    EmitRecord --> FilterPass{"RedactingFilter"}
    FilterPass --> ScrubDict["Scrub API keys & secrets from extra fields"]
    ScrubDict --> ScrubMsg["Regex scrub AIza..., sk-..., Bearer tokens"]
    ScrubMsg --> FormatterChoice{"Format Selection"}

    FormatterChoice -->|JSON| JsonFmt["JsonFormatter: Output structured JSON line"]
    FormatterChoice -->|Text| TextFmt["TextFormatter: Human-readable timestamp + msg"]

    JsonFmt --> TargetFile["Caller-configured log_file"]
    TextFmt --> TargetConsole["Standard Output"]
```

---

## 4. EmbeddingService Execution Flow

`EmbeddingService` provides standardized vector generation across local, OpenAI, and Google providers, with strict dimension validation, internal batching, token bucket rate limiting, and bounded retries:

```mermaid
flowchart TD
    Client[Caller / Application] -->|embed / embed_query / embed_documents| EmbeddingService

    subgraph InitPhase ["Initialization & Dimension Validation"]
        EmbeddingService --> InitService["Initialize: provider, model, dimensions, batch_size"]
        InitService --> ResolveProvider["Resolve Local, OpenAI, or Google Provider"]
        ResolveProvider --> ValDim{"provider.validate_dimensions(dimensions)"}
        ValDim -->|Mismatch| RaiseDimErr["Raise EmbeddingDimensionError"]
        ValDim -->|Supported| Ready["Service Ready"]
    end

    subgraph InputValidation ["Request & Input Validation"]
        Ready --> ReceiveReq["Receive EmbeddingRequest or kwargs"]
        ReceiveReq --> CheckEmpty{"Input empty or whitespace?"}
        CheckEmpty -->|Yes: input=[] or whitespace| RaiseValErr["Raise ValueError"]
        CheckEmpty -->|No| CheckBatchSize{"batch_size <= 0?"}
        CheckBatchSize -->|Yes| RaiseBatchErr["Raise ValueError"]
        CheckBatchSize -->|Valid| SetDeadline["Calculate overall request_deadline"]
        SetDeadline --> LogStart["safe_log: EMBEDDING_REQUEST_START"]
    end

    subgraph BatchingLoop ["Internal Batch Slicing & Resilient Execution"]
        LogStart --> SliceBatch["Slice inputs into chunks of batch_size"]
        
        SliceBatch --> RateLimitCheck["await rate_limiter.acquire(requests=1, tokens=est)"]
        RateLimitCheck -->|Throttled| LogRateLimit["safe_log: EMBEDDING_RATE_LIMITED"]
        RateLimitCheck -->|Capacity Ready| ExecWithRetry["execute_with_retry(provider.embed_batch, retry_policy)"]
        
        subgraph ProviderExecution ["Provider-Specific Dispatch"]
            ExecWithRetry --> RouteProvider{Active Provider}
            
            RouteProvider -->|Local| LocalExec["asyncio.to_thread(model.encode, ...)"]
            RouteProvider -->|OpenAI| OpenAIExec["await client.embeddings.create(...)"]
            RouteProvider -->|Google| GoogleExec["await client.aio.models.embed_content(...)"]
        end

        ProviderExecution --> Retried{"Transient 429/5xx or timeout?"}
        Retried -->|Yes: within retry limit| BackoffSleep["Wait with exponential backoff & jitter"]
        BackoffSleep --> LogRetry["safe_log: EMBEDDING_RETRY"]
        LogRetry --> ExecWithRetry
        Retried -->|No: Permanent 400/401/quota or deadline reached| RaiseError["Raise typed Error / Timeout"]

        ProviderExecution --> SuccessBatch["Return batch float vectors"]
        SuccessBatch --> ValidateDim{"Vector count & dim match?"}
        ValidateDim -->|No| RaiseDimErr2["Raise EmbeddingDimensionError"]
        ValidateDim -->|Yes| AppendResult["Append vectors in exact input order"]
        
        AppendResult --> RecordUsage["Aggregate provider usage: prompt_tokens, total_tokens"]
        RecordUsage --> MoreBatches{"More batches remaining?"}
        MoreBatches -->|Yes| SliceBatch
        MoreBatches -->|No| FinishBatches["All inputs processed"]
    end

    subgraph ResponseConstruction ["Response & Observability"]
        FinishBatches --> CalcMetrics["Calculate latency_ms; set usage"]
        CalcMetrics --> LogSuccess["safe_log: EMBEDDING_REQUEST_SUCCESS"]
        LogSuccess --> BuildResp["Return EmbeddingResponse"]
        BuildResp --> Client
    end

    subgraph ErrorHandling ["Error Shielding & Observability"]
        RaiseError -.-> LogFail["safe_log: EMBEDDING_TIMEOUT or EMBEDDING_REQUEST_FAILED"]
        LogFail --> PropagateErr["Propagate typed Exception to caller"]
    end
```

### 4.1 Service Initialization & Strict Dimension Validation
- **Location**: [`packages/genai-core/src/genai_core/services/embedding_service.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/services/embedding_service.py)
- **Execution Flow**:
  1. Resolves default configurations from `Settings` (`default_embedding_provider="local"`, `default_embedding_model="all-MiniLM-L6-v2"`, `default_embedding_dimensions=384`, `default_embedding_batch_size=32`).
  2. Resolves active provider: accepts registered provider dictionary, provider name string (`"local"`, `"openai"`, `"google"`), or direct `EmbeddingProvider` instance.
  3. **Strict Dimension Validation**: Calls `self.active_provider.validate_dimensions(self.dimensions)` at initialization.
     - For `LocalEmbeddingProvider`: `all-MiniLM-L6-v2` strictly rejects `dimensions=768` by raising `EmbeddingDimensionError(model='all-MiniLM-L6-v2', requested=768, supported=384)`.
     - For `OpenAIEmbeddingProvider`: `text-embedding-3-small` allows dimensions `<= 1536`, `text-embedding-3-large` allows `<= 3072`, while `text-embedding-ada-002` strictly requires `1536`.
     - For `GoogleEmbeddingProvider`: `text-embedding-005` native dimension is `768`, allowing output dimensionality `<= 768`. Prohibits retired `text-embedding-004`.
     - Invalid configurations fail fast before any inference or data processing occurs.

### 4.2 Request Preparation & Input Validation
- Accepts an [`EmbeddingRequest`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/models/embedding_model/embedding_request.py) object or keyword arguments (`input`, `batch_size`, `normalize_embeddings`).
- **Empty Input Checking**:
  - `input=[]`: raises `ValueError("Input cannot be empty.")`.
  - `input=""` or all-whitespace string: raises `ValueError("Input text cannot be empty or whitespace only.")`.
  - `input=["valid", "   "]`: raises `ValueError("Input text at index 1 cannot be empty or whitespace only.")`.
- **Batch Size Checking**:
  - `batch_size <= 0`: raises `ValueError("batch_size must be greater than 0.")`.
- **Bounded Request Deadline**: Computes overall request deadline `time.monotonic() + self.retry_policy.timeout_seconds` across the entire multi-batch request.
- Emits `EMBEDDING_REQUEST_START` via `safe_log` with provider, model, dimensions, total inputs, and batch counts.

### 4.3 Provider Adapters
1. **`LocalEmbeddingProvider`**:
   - Runs `sentence-transformers` via `asyncio.to_thread` for non-blocking execution.
   - Caches models using `@lru_cache(maxsize=4)`.
   - `sentence-transformers` is an optional extra (`genai-core[local]`).
2. **`OpenAIEmbeddingProvider`**:
   - Uses `AsyncOpenAI` SDK (`client.embeddings.create(input=texts, model=..., dimensions=...)`).
   - Sorts returned embeddings by item index to guarantee input order preservation regardless of API concurrency.
   - Stores exact token usage in `last_usage = {"prompt_tokens": ..., "total_tokens": ...}`.
3. **`GoogleEmbeddingProvider`**:
   - Uses current `google-genai` SDK with model `text-embedding-005`.
   - Configures `EmbedContentConfig(output_dimensionality=dimensions)` for variable dimensionality.
   - L2-normalizes vectors and wraps provider errors in `EmbeddingProviderError`.

---

## 5. Shared Resilience Execution Flow

Resilience logic is isolated in `packages/genai-core/src/genai_core/resilience/` as a shared capability reusable across services:

```mermaid
flowchart TD
    Operation[Async Service Operation: Batch Call or Store Search] --> LimiterAcquire["RateLimiter.acquire(requests=1, tokens=est_tokens)"]

    subgraph TokenBucketLimiter ["AsyncTokenBucket Rate Limiter (Per-Process)"]
        LimiterAcquire --> CheckHold{"In dynamic Retry-After hold?"}
        CheckHold -->|Yes| HoldWait["asyncio.sleep(remaining_hold)"]
        CheckHold -->|No| CheckBuckets{"RPM and TPM tokens available?"}
        HoldWait --> CheckBuckets
        CheckBuckets -->|Insufficient| CalcWait["Calculate required sleep to replenish"]
        CalcWait --> BucketWait["asyncio.sleep(wait_seconds)"]
        BucketWait --> ConsumeTokens["Consume 1 request token & N token capacity"]
        CheckBuckets -->|Available| ConsumeTokens
    end

    ConsumeTokens --> ExecRetry["execute_with_retry(operation, policy, deadline)"]

    subgraph RetryLoop ["Bounded Retry & Timeout Engine"]
        ExecRetry --> CheckDeadline{"Now >= Deadline?"}
        CheckDeadline -->|Yes| RaiseTimeout["Raise TimeoutError / Timeout Exception"]
        CheckDeadline -->|No| RunAttempt["await asyncio.wait_for(op(), attempt_timeout)"]
        
        RunAttempt -->|Success| ReturnResult["Return operation result"]
        RunAttempt -->|Timeout| AttemptTimedOut["Attempt timeout exceeded"]
        RunAttempt -->|Exception| EvalErr{"is_retryable_error(exc)?"}
        
        EvalErr -->|Non-Retryable: 400/401/quota/dim error| FastFail["Raise Exception immediately (no retry)"]
        AttemptTimedOut --> CheckMaxRetries{"attempt >= max_retries?"}
        EvalErr -->|Retryable: 429, 5xx, network error| CheckMaxRetries
        
        CheckMaxRetries -->|Exhausted| ReRaise["Re-raise last exception"]
        CheckMaxRetries -->|Attempts remain| ParseRetryAfter["extract_retry_after(exc)"]
        ParseRetryAfter --> NotifyLimiter["rate_limiter.record_rate_limit(wait)"]
        NotifyLimiter --> CalcBackoff["delay = min(max_backoff, init * 2^attempt) + jitter"]
        CalcBackoff --> SleepRetry["await asyncio.sleep(delay)"]
        SleepRetry --> NextAttempt["attempt += 1"]
        NextAttempt --> CheckDeadline
    end
```

### 5.1 Per-Process Token Bucket Rate Limiting
- **Location**: [`packages/genai-core/src/genai_core/resilience/rate_limiter.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/resilience/rate_limiter.py)
- **Mechanics**:
  - Maintained as dual token buckets: Request Bucket (RPM) and Token Bucket (TPM).
  - Continuous fractional replenishment based on elapsed wall-clock time (`time.monotonic()`).
  - Thread/coroutine-safe asynchronous acquisition via `asyncio.Lock`.
  - Non-blocking waiting: coroutines sleep only for the exact duration required to replenish capacity, without blocking unrelated async tasks.
  - **Scope**: Documented as strictly **per-process in-memory**. Does not require Redis or external infrastructure in this phase.
  - **Internal Pacing Token Estimation**: Uses `estimate_tokens(texts)` heuristic (~4 characters per token) strictly for TPM pacing. These pacing estimates are internal and never leaked as actual usage in `EmbeddingResponse`.

### 5.2 Error Classification & Selective Retryability
- **Location**: [`packages/genai-core/src/genai_core/resilience/retry_policy.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/resilience/retry_policy.py)
- **Retryable Errors**:
  - Network timeouts, connection resets, transient DNS failures.
  - HTTP `408 Request Timeout`, `429 Too Many Requests`.
  - HTTP `500 Internal Server Error`, `502 Bad Gateway`, `503 Service Unavailable`, `504 Gateway Timeout`.
- **Non-Retryable Errors (Fail-Fast)**:
  - Input validation errors (`ValueError`, `TypeError`).
  - Strict dimension mismatches (`EmbeddingDimensionError`, `IncompatibleEmbeddingError`).
  - Unsupported search mode errors (`UnsupportedSearchModeError`).
  - Missing dependency errors (`MissingExtraDependencyError`).
  - HTTP 400 Bad Request, 401 Unauthorized, 403 Forbidden.
  - Quota exhaustion / billing failure (`insufficient_quota`, `billing_not_active`, `account_deactivated`).
- **`Retry-After` Header Parsing**: Parses integer seconds, floats, or RFC-2822 date strings from `Retry-After` HTTP headers. When received on a 429 response, informs the `RateLimiter` to pause subsequent dispatches dynamically.

---

## 6. RetrievalService, Hybrid Search & Re-ranking Flow

`RetrievalService` provides unified semantic (dense), full-text (sparse), and hybrid search, score fusion, optional cross-encoder re-ranking, and document indexing:

```mermaid
flowchart TD
    Client[Caller / Application] -->|search / index_documents / delete_documents| RetrievalService

    subgraph ServiceInit ["Service Initialization"]
        RetrievalService --> InitService["Initialize: store, embedding_service, reranker, retry_policy"]
        InitService --> CheckStoreDim{"embedding_service != None?"}
        CheckStoreDim -->|Yes| CompareDims{"embedding_service.dimensions == store.dimensions?"}
        CompareDims -->|Mismatch| RaiseIncompat["Raise IncompatibleEmbeddingError"]
        CompareDims -->|Match| Ready["Service Ready"]
        CheckStoreDim -->|No| Ready
    end

    subgraph SearchFlow ["search(query) Execution Path"]
        Ready --> ReceiveSearch["Receive RetrievalQuery or query string"]
        ReceiveSearch --> CheckMode{"query.search_mode?"}

        CheckMode -->|dense| ExecDense["Dense Path: resolve query_vector -> store.search()"]
        CheckMode -->|sparse| ExecSparse["Sparse Path: keyword match -> store.search_keyword()"]
        CheckMode -->|hybrid| ExecHybrid["Hybrid Path: asyncio.gather(vector_search, keyword_search)"]

        ExecHybrid --> ScoreFusion["weighted_score_fusion: min-max normalization + weighted sum"]
        ExecDense --> CandidatesReady["Candidate results ready"]
        ExecSparse --> CandidatesReady
        ScoreFusion --> CandidatesReady

        CandidatesReady --> CheckRerank{"should_rerank == True?"}
        CheckRerank -->|No| SliceTopK["Slice to top_k"]
        CheckRerank -->|Yes| CheckRerankerAvail{"reranker configured?"}

        CheckRerankerAvail -->|No & warn_and_continue| WarnFallback["Set rerank_message, return candidates"]
        CheckRerankerAvail -->|No & raise| RaiseRerankErr["Raise RetrievalError"]
        CheckRerankerAvail -->|Yes| ExecRerank["await reranker.rerank(query, candidates, top_k)"]
        ExecRerank --> SetReranked["Preserve retrieval_score in metadata, set reranked=True"]

        SliceTopK --> CalcLatency["Calculate latency_ms"]
        WarnFallback --> CalcLatency
        SetReranked --> CalcLatency

        CalcLatency --> LogSearchSuccess["safe_log: RETRIEVAL_SUCCESS"]
        LogSearchSuccess --> BuildResp["Return RetrievalResponse (results, reranked, metrics)"]
        BuildResp --> Client
    end

    subgraph IndexFlow ["index_documents(documents) Execution Path"]
        Ready --> ReceiveIndex["Receive list[DocumentChunk] or list[str]"]
        ReceiveIndex --> CheckDocType{"Elements are str or DocumentChunk?"}

        CheckDocType -->|str| EmbedRaw["await embedding_service.embed_documents(texts)"]
        EmbedRaw --> BuildChunks["Construct DocumentChunk with generated IDs and vectors"]
        BuildChunks --> StoreUpsert["await store.upsert(chunks)"]

        CheckDocType -->|DocumentChunk| CheckMissingVecs{"Any chunk.embedding is None?"}
        CheckMissingVecs -->|Yes| EmbedMissing["await embedding_service.embed_documents(missing_texts)"]
        EmbedMissing --> AssignMissing["Assign generated vectors to chunks"]
        AssignMissing --> StoreUpsert
        CheckMissingVecs -->|No| StoreUpsert

        StoreUpsert --> ReturnCount["Return count of upserted chunks"]
        ReturnCount --> Client
    end

    subgraph ErrorHandling ["Error Handling & Timeout Protection"]
        ExecStoreSearch -.->|Timeout / Deadline| LogFail["safe_log: RETRIEVAL_FAILED"]
        LogFail --> RaiseTimeoutErr["Raise RetrievalTimeoutError"]
        ExecStoreSearch -.->|Exception| LogFail2["safe_log: RETRIEVAL_FAILED"]
        LogFail2 --> ReRaise["Re-raise exception"]
    end
```

### 6.1 Query Validation and Extensibility Rules
- **Location**: [`packages/genai-core/src/genai_core/models/retrieval_model/retrieval_query.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/models/retrieval_model/retrieval_query.py)
- **Input Validation**:
  - `query` (text) or `query_vector` (float list) must be provided. If both are omitted, empty, or whitespace-only, raises `ValueError`.
  - `top_k` must be greater than 0 (`top_k <= 0` raises `ValueError`).
  - `score_threshold` if provided must be bounded between `0.0` and `1.0`.
- **Search Mode Policy**:
  - The request contract defines `search_mode: Literal["dense", "sparse", "hybrid"] = "dense"`.
  - In Phase 3, **dense search is exclusively supported**.
  - Requesting `"sparse"` or `"hybrid"` strictly raises `UnsupportedSearchModeError(mode)` explaining that sparse and hybrid retrieval are planned for a subsequent stage. The system never silently falls back or pretends hybrid retrieval is operating.

### 6.2 Embedding Compatibility and Dimension Enforcement
- **Location**: [`packages/genai-core/src/genai_core/services/retrieval_service.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/services/retrieval_service.py)
- At service initialization, if an `EmbeddingService` is provided, `self.embedding_service.dimensions` is validated against `self.store.dimensions`. A mismatch immediately raises `IncompatibleEmbeddingError(requested, supported)`.
- When pre-computed `query_vector` is supplied in `RetrievalQuery`, `len(query_vector)` is validated against `self.store.dimensions`.

### 6.3 In-Memory Vector Store (`InMemoryVectorStore`)
- **Location**: [`packages/genai-core/src/genai_core/retrieval/in_memory_store.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/retrieval/in_memory_store.py)
- Implements [`VectorStoreAdapter`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/retrieval/protocol.py).
- Zero external dependencies: computes exact cosine similarity `dot(a, b) / (norm(a) * norm(b))`, clamped to `[-1.0, 1.0]`.
- Supports exact metadata key-value filtering, score threshold filtering, and descending rank assignment.
- Enables deterministic, fast, offline unit testing without requiring Docker or PostgreSQL.

### 6.4 PostgreSQL pgvector Store (`PGVectorStoreAdapter`)
- **Location**: [`packages/genai-core/src/genai_core/retrieval/pgvector_store.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/retrieval/pgvector_store.py)
- **Role**: Concrete database adapter implementing [`VectorStoreAdapter`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/retrieval/protocol.py) with PostgreSQL and `pgvector`.
- **Injected Connection Management**: Requires injected `connection`, `connection_pool` (e.g. `psycopg_pool.AsyncConnectionPool`), or `connection_factory`. Never creates ad-hoc unpooled TCP connections per operation.
- **SQL Safety**:
  - Whitelist regex validation (`^[a-zA-Z_][a-zA-Z0-9_]*$`) for all table and column names, preventing SQL injection via dynamic identifiers.
  - Strict parameterization using `%s` placeholders for all dynamic inputs (vectors, filters, thresholds, chunk IDs).
- **Cosine Similarity & Ranking**:
  - Uses pgvector cosine distance operator `<=>`: `1 - (embedding <=> %s) AS similarity_score`.
  - Orders by `embedding <=> %s ASC` (nearest neighbors first) with `LIMIT %s`.
- **Metadata Filtering**:
  - Utilizes PostgreSQL JSONB containment (`metadata @> %s::jsonb`) to support dictionary subset filtering.
  - Supports optional dedicated tenant column (`tenant_id_column`) for indexed tenant partition filtering (`WHERE tenant_id = %s`).
- **Upsert & Delete**:
  - Upsert: Parameterized batch `INSERT INTO ... ON CONFLICT ({conflict_target}) DO UPDATE SET ...` updating text, vector, and metadata.
  - Delete: Parameterized `DELETE FROM {table_name} WHERE {id_col} = ANY(%s)`.
- **Resilience**: Translates database errors into `VectorStoreError`. Flags transient network/operational errors (connection reset, timeout, broken pipe, deadlock) as `retryable=True` for automatic retry by `execute_with_retry`.
- **Optional Dependency**: Declared under `[project.optional-dependencies] postgres = ["psycopg[binary]>=3.1.0", "pgvector>=0.2.0"]`. Attempting to use the adapter without installed dependencies raises `MissingExtraDependencyError`.

### 6.5 Document Indexing and Provenance
- `index_documents` accepts raw text strings or structured `DocumentChunk` instances.
- If raw strings or chunks lacking vectors are provided, `RetrievalService` utilizes `self.embedding_service.embed_documents` to generate embeddings in batches before dispatching to `store.upsert()`.
- Search results preserve the original `metadata`, `source_id`, and `rank` in [`RetrievalResult`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/models/retrieval_model/retrieval_response.py), guaranteeing evidence traceability for downstream prompt generation and citations.

---

## 7. Future Services (Planned Architecture)

As `genai-core` expands in Modules 17–20, additional shared capabilities will connect into the package architecture:

### 7.1 PostgreSQL pgvector Integration (Stage 3) — Completed
- **Location**: [`packages/genai-core/src/genai_core/retrieval/pgvector_store.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/retrieval/pgvector_store.py)
- **Status**: Completed and fully verified with 14 dedicated unit tests.

### 7.2 Hybrid Retrieval & Reranking (Phase 3 Follow-Up)
- **Location**: `genai_core.retrieval.hybrid`
- **Role**: Combines dense vector retrieval with PostgreSQL full-text search (`ts_rank_cd`), applying min-max score normalization and reciprocal rank or weighted score fusion, followed by optional cross-encoder reranking.

### 7.3 Memory Primitives (Phase 4)
- **Location**: `genai_core.memory`
- **Role**: Session and user scoped conversation memory, buffer windows, and persistent store adapters.
- **Interconnection**: Injects conversation history into `LLMService` requests (`ChatMessage` lists) and extracts facts/summaries post-completion.

### 7.4 Tool & Agent Orchestration (Phase 5)
- **Location**: `genai_core.agents` / `genai_core.tools`
- **Role**: Higher-level multi-step tool execution loops, argument dispatching, and agent state machines.
- **Interconnection**: Dispatches tool definitions (`ToolSpec`) to `LLMService`, executes validated functions under caller control, and reports tool execution steps via `safe_log`.

---

## 8. Verification and Test Flow

The behavior of `genai-core` is verified through automated unit tests organized by feature under `tests/unit/` and integration tests under `tests/integration/`:
- **Retrieval unit tests (`tests/unit/retrieval/`)**:
  - [`test_fusion.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/unit/retrieval/test_fusion.py) validates min-max normalization (empty sets, singletons, identical scores, diverse scores), weighted score blending with custom weights, disjoint vs overlapping candidate handling, and score threshold filtering.
  - [`test_reranker.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/unit/retrieval/test_reranker.py) validates protocol conformance (`isinstance(reranker, Reranker)`), empty candidate handling, missing optional `sentence-transformers` dependency raising `MissingExtraDependencyError`, cross-encoder prediction, and preservation of initial scores in `metadata['retrieval_score']`.
  - [`test_retrieval_service.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/unit/retrieval/test_retrieval_service.py) validates dense semantic search, sparse keyword search (skipping embeddings), hybrid search with concurrent execution and score fusion, optional cross-encoder re-ranking with candidate expansion, fallback behavior (`warn_and_continue` vs `raise`), store timeout handling, document indexing (strings and chunks), and observability events.
  - [`test_in_memory_store.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/unit/retrieval/test_in_memory_store.py) validates protocol conformance (`isinstance(store, VectorStoreAdapter)`), dimension validation, upsert validation, chunk updates, chunk deletion, cosine vector ranking, full-text token keyword matching (`search_keyword`), metadata key-value filtering, and score thresholding.
  - [`test_pgvector_store.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/unit/retrieval/test_pgvector_store.py) validates protocol conformance (`isinstance(store, VectorStoreAdapter)`), SQL identifier safety against injection, vector search and full-text keyword search (`ts_rank_cd` and `websearch_to_tsquery`), dedicated vs fallback `to_tsvector` column paths, parameterization of query values, metadata JSONB filtering, score thresholding, batch upsert with `ON CONFLICT`, delete execution with `ANY(%s)`, transient vs permanent failure classification, missing dependency raising `MissingExtraDependencyError`, and end-to-end integration with `RetrievalService`.
- **Embedding unit tests (`tests/unit/embedding/`)**:
  - [`test_embedding_service.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/unit/embedding/test_embedding_service.py) validates initialization defaults, strict dimension rejection, empty input rejection, batch size validation, internal slicing loop, 1-to-1 order preservation, missing dependency handling, convenience helpers, and observability events.
  - [`test_embedding_resilience.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/unit/embedding/test_embedding_resilience.py) validates RPM/TPM token bucket pacing, token estimation heuristics, error classification, exponential backoff, jitter, Retry-After header parsing, attempt timeout, deadline enforcement, and service retry integration.
  - [`test_openai_embedding_provider.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/unit/embedding/test_openai_embedding_provider.py) validates dimension validation, request creation, output normalization, index-ordered sorting, usage tracking, and error wrapping.
  - [`test_google_embedding_provider.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/unit/embedding/test_google_embedding_provider.py) validates `text-embedding-005` model configuration, output dimensionality, embedding extraction, and error wrapping.
- **LLM unit tests (`tests/unit/llm/`)**:
  - [`test_llm_service.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/unit/llm/test_llm_service.py) validates dispatch, error shielding, `complete_all` concurrency, and stream delegation.
  - [`test_protocol.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/unit/llm/test_protocol.py) validates provider adapters conform to `LLMProvider`.
  - [`test_tool_spec.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/unit/llm/test_tool_spec.py) validates schema generators and argument parsing.
  - [`test_google_provider.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/unit/llm/test_google_provider.py), [`test_openai_provider.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/unit/llm/test_openai_provider.py), [`test_anthropic_provider.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/unit/llm/test_anthropic_provider.py).
- **Observability unit tests (`tests/unit/observability/`)**:
  - [`test_observability.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/unit/observability/test_observability.py) validates caller-defined destinations, JSON/text formatting, secret scrubbing, async request context, idempotency, and error shielding.
- **Live integration tests (`tests/integration/`)**:
  - **LLM integration tests (`tests/integration/llm/`)**: [`test_anthropic_provider_integration.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/integration/llm/test_anthropic_provider_integration.py), [`test_google_provider_integration.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/integration/llm/test_google_provider_integration.py), [`test_openai_provider_integration.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/integration/llm/test_openai_provider_integration.py).
  - **Embedding integration tests (`tests/integration/embedding/`)**: [`test_local_embedding_integration.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/integration/embedding/test_local_embedding_integration.py) (validates real model inference, vector dimensions, unit normalization, and semantic distance), [`test_google_embedding_integration.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/integration/embedding/test_google_embedding_integration.py), [`test_openai_embedding_integration.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/integration/embedding/test_openai_embedding_integration.py).
  - **Retrieval integration tests (`tests/integration/retrieval/`)**: [`test_cross_encoder_reranker_integration.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/integration/retrieval/test_cross_encoder_reranker_integration.py) (validates live neural re-ranking, candidate scoring, and retrieval score tracking), [`test_pgvector_store_integration.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/integration/retrieval/test_pgvector_store_integration.py) (validates vector search, full-text keyword matching, metadata JSONB filtering, and deletion against PostgreSQL), [`test_hybrid_retrieval_integration.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/integration/retrieval/test_hybrid_retrieval_integration.py) (validates end-to-end pipeline combining embeddings, vector store, fusion, and re-ranking).
  - **Observability integration tests (`tests/integration/observability/`)**: [`test_observability_integration.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/integration/observability/test_observability_integration.py).

