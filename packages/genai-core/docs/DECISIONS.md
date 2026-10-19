# Architecture & Implementation Decisions: `genai-core` Scope

This document records the architectural decisions, component responsibilities, design patterns, dependencies, and rationale for `genai-core` (`packages/genai-core/`).

---

## 2026-10-16: Architecture and Design of `genai-core`

### 1. Unified `LLMProvider` Protocol
- **Location**: [`packages/genai-core/src/genai_core/llm/llm_provider.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/llm/llm_provider.py)
- **Decision**: Define a structural typing interface using `typing.Protocol` with `@runtime_checkable`, containing three core methods:
  ```python
  class LLMProvider(Protocol):
      async def complete_structure(
          self, prompt: str, response_schema: type[BaseModel] | None = None
      ) -> LLMResponseModel: ...

      async def complete_tool(
          self,
          messages: list[ChatMessage],
          tools: list[ToolSpec] | None = None,
          tool_choice: ToolChoice | None = None,
      ) -> LLMResponseModel: ...

      def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]: ...
  ```
- **Rationale**:
  - Unifies structured output generation (formerly prompt/schema based) and multi-turn tool calling (message/tool based) behind a single provider contract without forcing either mode into an unnatural signature.
  - Adding `@runtime_checkable` enables validation in tests via `isinstance(provider, LLMProvider)`.
  - Providers implement this protocol implicitly without strict subclassing requirements.

### 2. Normalized Unified Response Model (`LLMResponseModel`)
- **Location**: [`packages/genai-core/src/genai_core/models/llm_model/llm_response_model.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/models/llm_model/llm_response_model.py)
- **Decision**: Unify text generation, structured data output, and tool invocation metadata into one standardized model:
  - `text: str | None = None`
  - `data: dict[str, Any] | None = None` (holds parsed JSON structure when `response_schema` is requested)
  - `tool_calls: list[ToolCall] = Field(default_factory=list)` (holds parsed function calls)
  - `model: str`
  - `latency_ms: float`
  - `input_tokens: int = 0`
  - `output_tokens: int = 0`
  - `raw_response: Any | None = None`
- **Rationale**:
  - Downstream callers (services, agents, tools) receive a single normalized response object regardless of whether Google Gemini, OpenAI, or Anthropic Claude served the request.
  - Both structured schema outputs (`data`) and tool invocations (`tool_calls`) coexist cleanly.

### 3. Separation of Concerns & Dependency Injection in Provider Adapters
- **Location**:
  - [`packages/genai-core/src/genai_core/llm/google_provider.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/llm/google_provider.py)
  - [`packages/genai-core/src/genai_core/llm/openai_provider.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/llm/openai_provider.py)
  - [`packages/genai-core/src/genai_core/llm/anthropic_provider.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/llm/anthropic_provider.py)
- **Decision**: Keep provider-specific SDK imports (`google.genai`, `openai`, `anthropic`), SDK options, and payload mappings strictly isolated inside their respective adapter classes.
- **Dependency Injection**:
  - Each provider adapter's `__init__` accepts an optional `client`, `model`, `temperature`, and `api_key`.
  - When `client` is provided (e.g., in unit tests), no external network calls or active API keys are required.
  - Fallback settings are retrieved via `get_settings()`.

### 4. Provider Implementation Characteristics & Known Capabilities
- **Google Gemini (`GoogleProvider`)**:
  - Implements native structured schema enforcement using `GenerateContentConfig(response_mime_type="application/json", response_schema=response_schema)`.
  - Converts `ToolSpec` to `google.genai.types.FunctionDeclaration`.
  - Requires `tool_name` on tool response messages (enforces Google Gemini API requirement).
  - Handles streaming via `client.aio.models.generate_content_stream`.
- **OpenAI (`OpenAIProvider`)**:
  - Implements structured output via `text={"format": {"type": "json_schema", ...}}` on `responses.create`.
  - Implements tool calls via `FunctionToolParam` and `ToolChoice`.
  - Implements streaming via `client.responses.stream`.
- **Anthropic Claude (`AnthropicProvider`)**:
  - Implements tool calling via `input_schema` and `tool_choice={"type": "tool", "name": ...}`.
  - For `complete_structure`, Anthropic executes `messages.create` and attempts JSON decoding of the returned text block into `data`. (Anthropic does not expose a native constrained-grammar JSON schema parameter on standard text generation).
  - Implements streaming via `client.messages.stream`.

### 5. `LLMService` Entry Point and Error Shielding
- **Location**: [`packages/genai-core/src/genai_core/services/llm_services.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/services/llm_services.py)
- **Decision**:
  - `LLMService` acts as the single provider-independent orchestration point.
  - Accepts a `providers: dict[str, LLMProvider] | None` dictionary for dependency injection.
  - Default initialization (`_init_default_providers`) catches missing API keys/client errors to prevent startup crashes when only specific providers are configured.
  - Catches provider exceptions at the boundary and translates them into an `LLMManagerResponse(error=LLMError(...))` with extracted status codes, error codes, and retryability flags (`408, 429, 500, 502, 503, 504`).
  - Supports concurrent execution across all registered providers via `complete_all()`.

### 6. Backward Compatibility Re-Exports and Aliases
- **Locations**:
  - [`packages/genai-core/src/genai_core/core/__init__.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/core/__init__.py) -> aliases `config`
  - [`packages/genai-core/src/genai_core/providers/__init__.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/providers/__init__.py) -> aliases `llm`
  - [`packages/genai-core/src/genai_core/services/llm_services.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/services/llm_services.py) -> `LLMServicemanager = LLMService`
- **Decision**: Maintain aliases so callers can import from `genai_core.llm` or `genai_core.providers`, `genai_core.config` or `genai_core.core`, and use either `LLMService` or `LLMServicemanager`.

### 7. Dual Error Paradigms
- **Locations**:
  - [`packages/genai-core/src/genai_core/models/llm_model/llm_response_model.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/models/llm_model/llm_response_model.py) (`class LLMError(BaseModel)`)
  - [`packages/genai-core/src/genai_core/exceptions/llm_exceptions.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/exceptions/llm_exceptions.py) (`class LLMError(Exception)`)
- **Decision**: Provide the Pydantic `LLMError` model for serialized service responses, while maintaining explicit Python exception classes (`LLMAuthenticationError`, `LLMRateLimitError`, etc.) in `exceptions/llm_exceptions.py` for direct exception handling. In `genai_core/__init__.py`, the exception base is exported as `LLMException` to prevent naming collisions with the model `LLMError`.

### 8. Reserved Directories
- **Locations**:
  - `packages/genai-core/src/genai_core/agents/`
  - `packages/genai-core/src/genai_core/embedding/`
  - `packages/genai-core/src/genai_core/evaluation/`
  - `packages/genai-core/src/genai_core/memory/`
  - `packages/genai-core/src/genai_core/observability/`
  - `packages/genai-core/src/genai_core/prompts/`
  - `packages/genai-core/src/genai_core/retrieval/`
  - `packages/genai-core/src/genai_core/tools/`
  - `packages/genai-core/tests/integration/`
- **Status**: Kept strictly as empty structural placeholders. No agent, memory, evaluation, or embedding logic is implemented to keep package scope focused on the LLM core abstraction.

---

## 2026-10-16: Structured Output Validation, Retries, and `structured_response`

### 1. Renaming `data` to `structured_response` with Backward Compatibility
- **Locations**:
  - [`packages/genai-core/src/genai_core/models/llm_model/llm_response_model.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/models/llm_model/llm_response_model.py)
  - [`packages/genai-core/src/genai_core/models/llm_model/llm_manager_response.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/models/llm_model/llm_manager_response.py)
- **Decision**: Replace `data: dict[str, Any] | None` with `structured_response: Any | None = None`. Provide a `@property def data(self)` alias on both models returning `self.structured_response`.
- **Rationale**: `structured_response` explicitly describes that the field holds the validated Pydantic model instance produced by `complete_structure`. The property alias guarantees that any existing code expecting `.data` continues to function.

### 2. Pydantic Schema Validation in `complete_structure`
- **Locations**:
  - [`packages/genai-core/src/genai_core/llm/google_provider.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/llm/google_provider.py)
  - [`packages/genai-core/src/genai_core/llm/openai_provider.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/llm/openai_provider.py)
  - [`packages/genai-core/src/genai_core/llm/anthropic_provider.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/llm/anthropic_provider.py)
- **Decision**: After parsing JSON from model output, validate it via `response_schema.model_validate(raw_data)`. Return the validated instance in `structured_response` instead of raw dictionary.
- **Rationale**: Guarantees type safety and validation constraints enforced by the user's Pydantic schema.

### 3. Automatic 2-Retry Mechanism with Error Feedback
- **Decision**: When `complete_structure` encounters a JSON decode error or Pydantic `ValidationError`:
  - Perform up to 2 retries (total 3 attempts).
  - On each retry, inject an error feedback prompt informing the model of its previous output and the exact validation error details, instructing it to fix the output.
  - If all retries fail, raise `LLMValidationError` containing the failure message and raw output.
- **Rationale**: LLMs frequently produce minor formatting deviations that can be self-corrected when presented with explicit schema validation feedback.

### 4. `LLMValidationError` and Error Shielding in `LLMService`
- **Locations**:
  - [`packages/genai-core/src/genai_core/exceptions/llm_exceptions.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/exceptions/llm_exceptions.py)
  - [`packages/genai-core/src/genai_core/services/llm_services.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/services/llm_services.py)
- **Decision**: Define `LLMValidationError` as a distinct exception. `LLMService.complete` catches `LLMValidationError` and converts it into `LLMManagerResponse(error=LLMError(code="validation_error", status_code=422, retryable=False))`.

---

## 2026-10-16: Full Removal of `.data`, Migration of Settings to `core`, and Configurable Retries

### 1. Complete Removal of `.data` Name
- **Locations**:
  - [`packages/genai-core/src/genai_core/models/llm_model/llm_response_model.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/models/llm_model/llm_response_model.py)
  - [`packages/genai-core/src/genai_core/models/llm_model/llm_manager_response.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/models/llm_model/llm_manager_response.py)
  - [`packages/genai-core/src/genai_core/llm/`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/llm/)
  - Unit tests in `packages/genai-core/tests/unit/`
- **Decision**: Completely removed the `.data` field and `@property def data` backward-compatibility aliases. The structured output field is named exclusively `structured_response`. In internal provider parsing, renamed all intermediate `raw_data` variables to `parsed_json` to avoid confusion.
- **Rationale**: User explicitly directed to avoid any aliasing with `data` and ensure `structured_response` is the single source of truth across the entire package.

### 2. Removal of `config/` Directory and Consolidation into `core/`
- **Locations**:
  - Removed: `packages/genai-core/src/genai_core/config/`
  - Canonical: [`packages/genai-core/src/genai_core/core/settings.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/core/settings.py)
  - [`packages/genai-core/src/genai_core/core/__init__.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/core/__init__.py)
  - [`packages/genai-core/src/genai_core/__init__.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/__init__.py)
- **Decision**: Removed the `config` folder completely. Placed `settings.py` directly inside `genai_core.core`. Re-exported `Settings` and `get_settings` from `genai_core.core` and `genai_core`.
- **Rationale**: Consolidates package configuration and foundational settings inside `core`, eliminating redundant top-level folders.

### 3. Configurable `max_structured_error_retry` in `Settings`
- **Locations**:
  - [`packages/genai-core/src/genai_core/core/settings.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/core/settings.py)
  - [`packages/genai-core/src/genai_core/llm/google_provider.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/llm/google_provider.py)
  - [`packages/genai-core/src/genai_core/llm/openai_provider.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/llm/openai_provider.py)
  - [`packages/genai-core/src/genai_core/llm/anthropic_provider.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/llm/anthropic_provider.py)
- **Decision**: Added `max_structured_error_retry: int = 2` to `Settings`. In `complete_structure`, if `max_retries` is not explicitly passed, providers default to `get_settings().max_structured_error_retry`.
- **Rationale**: Enables centralized configuration of structured error retry limits via environment variables (`MAX_STRUCTURED_ERROR_RETRY`) or settings overrides without requiring hardcoded limits.

---

## 2026-10-16: Live Integration Tests for All 3 LLM Providers

### 1. Dedicated Live Tests
- **Locations**:
  - [`packages/genai-core/tests/integration/test_google_provider_integration.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/integration/test_google_provider_integration.py)
  - [`packages/genai-core/tests/integration/test_openai_provider_integration.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/integration/test_openai_provider_integration.py)
  - [`packages/genai-core/tests/integration/test_anthropic_provider_integration.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/integration/test_anthropic_provider_integration.py)
- **Decision**: Created live integration tests covering `complete_structure`, `complete_tool`, and `stream` for Google Gemini, OpenAI, and Anthropic Claude.
- **Rationale**: Verifies real-world connectivity, schema adherence, and streaming mechanics directly with each provider.

### 2. Gating Execution on `INTEGRATION_TEST=1`
- **Locations**:
  - [`packages/genai-core/src/genai_core/core/settings.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/core/settings.py)
  - [`packages/genai-core/tests/conftest.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/conftest.py)
- **Decision**: Added `integration_test: int = 0` to `Settings`. Installed an `autouse` fixture in `conftest.py` that skips all `@pytest.mark.integration` tests unless `INTEGRATION_TEST=1` is specified in `.env` or the environment.
- **Rationale**: Prevents accidental live API billing and test failures in environments without configured credentials.

### 3. Graceful Error Handling for OpenAI Credits and Missing Anthropic Key
- **Decision**: Integration tests for OpenAI intercept quota exhaustion, billing errors, and rate limits (429 / `insufficient_quota`), cleanly skipping the test. Anthropic integration tests intercept missing keys, authorization errors (401/403), and credit issues, skipping without test suite failure. Google integration tests intercept transient 503 high-demand spike errors and skip gracefully.
- **Rationale**: Enables developers to run integration tests against available and funded providers without failing the entire suite for unconfigured or unfunded services.

---

## 2026-10-16: Reusable Caller-Controlled Observability for `genai-core`

### 1. Caller-Controlled Log Destination & Zero Side-Effects by Default
- **Locations**:
  - [`packages/genai-core/src/genai_core/observability/config.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/observability/config.py)
  - [`packages/genai-core/src/genai_core/observability/logger.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/observability/logger.py)
- **Decision**: Implemented `ObservabilityConfig` and `configure_observability()`. By default, `genai-core` initializes with a `logging.NullHandler()` and creates no files or directories. File logging is activated only when the consuming application explicitly supplies a file path (`log_file`), at which point the required parent directory is created and a `RotatingFileHandler` (or standard `FileHandler`) is attached.
- **Alternatives Considered**: Module 09's approach of creating a hardcoded `logs/rag_app.log` in `parents[3]` at module import time. Rejected because it produces filesystem pollution, conflicts with caller directory layouts, and violates library reusability principles.
- **Impact**: Consuming applications maintain full control over where log artifacts are stored, rotation limits, log formats, and stdout emission.

### 2. Automated Secret and Credential Redaction
- **Locations**:
  - [`packages/genai-core/src/genai_core/observability/redactor.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/observability/redactor.py)
  - [`packages/genai-core/src/genai_core/observability/formatters.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/observability/formatters.py)
- **Decision**: Added `RedactingFilter` and redaction utilities (`redact_string`, `redact_data`). Scrubs Google API keys (`AIza...`), OpenAI keys (`sk-...`), Anthropic keys (`sk-ant-...`), Bearer authorization headers, and credential dict keys (`api_key`, `secret`, `password`, `token`). Configured whitelist for non-sensitive token metrics (`input_tokens`, `output_tokens`, `total_tokens`, `token_count`) to avoid false-positive masking.
- **Alternatives Considered**: Relying on developers never logging secrets or using external masking libraries. Rejected to keep `genai-core` dependency-free and prevent accidental credential exposure in production or logs.
- **Impact**: All messages, extra attributes, and exception tracebacks emitted by `genai_core` are automatically sanitized before hitting disk or stdout.

### 3. Fault-Tolerant Logging & Safe Dispatch (`safe_log`)
- **Locations**:
  - [`packages/genai-core/src/genai_core/observability/logger.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/observability/logger.py)
  - [`packages/genai-core/src/genai_core/services/llm_services.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/services/llm_services.py)
- **Decision**: Wrapped all instrumentation points in `LLMService` (`complete` and `stream`) with `safe_log`. Logging failures (e.g. disk full, permission errors, serialization issues) are suppressed and never cause the primary LLM call to fail.
- **Alternatives Considered**: Module 09's approach of raising `RuntimeError` when `request_id` was absent. Rejected because logging should never crash business operations.
- **Impact**: Zero downtime or request failure caused by observability subsystem issues.

---

## 2026-10-17: Embedding Service Architecture (Stage 1 & Stage 2)

### 1. Default Provider & Optional Extra Dependency
- **Locations**:
  - [`packages/genai-core/pyproject.toml`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/pyproject.toml)
  - [`packages/genai-core/src/genai_core/embedding/local_provider.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/embedding/local_provider.py)
- **Decision**: The default embedding provider is `local` using `sentence-transformers/all-MiniLM-L6-v2`. `sentence-transformers` is declared as an optional package extra (`[project.optional-dependencies] local = ["sentence-transformers>=3.0.0"]`). When the dependency is absent, `LocalEmbeddingProvider` raises a typed `MissingExtraDependencyError` with instructions to install `genai-core[local]`.
- **Alternatives Considered**: Making `sentence-transformers` and PyTorch mandatory dependencies of base `genai-core`. Rejected because heavy ML libraries would force 2GB+ downloads on consumers that only need LLM API access or hosted embeddings.
- **Impact**: Keeps the base package lightweight and fast to install while providing high-performance local vector generation out of the box when the `local` extra is installed.

### 2. Strict Service-Level Dimension Validation
- **Locations**:
  - [`packages/genai-core/src/genai_core/services/embedding_service.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/services/embedding_service.py)
  - [`packages/genai-core/src/genai_core/exceptions/embedding_exceptions.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/exceptions/embedding_exceptions.py)
- **Decision**: Embedding output dimensions are configured and strictly validated during service initialization (`EmbeddingService(dimensions=...)`). Whenever the provider model's native dimension is known or discoverable, unsupported dimensions are immediately rejected with `EmbeddingDimensionError(model, requested, supported)`. Returned batch vector sizes are also verified.
- **Alternatives Considered**: Silent truncation, zero-padding, or postponing dimension checking until vectors are stored downstream in a vector database. Rejected because silently altering vector dimensions invalidates semantic cosine similarity and breaks downstream retrieval.
- **Impact**: Incompatible configurations fail fast at initialization before executing expensive model inference or data ingestion.

### 3. Clear Separation Between Batch Size and Vector Dimensions
- **Location**: [`packages/genai-core/src/genai_core/services/embedding_service.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/services/embedding_service.py)
- **Decision**: `batch_size` (number of texts processed per forward pass, default 32) and `dimensions` (vector length, e.g. 384) are completely separated in configuration and method signatures. Batch size <= 0 raises `ValueError`.
- **Alternatives Considered**: Conflating batch size and model dimension into a generic "shape" parameter. Rejected for clarity and usability.
- **Impact**: Eliminates developer confusion between batch slicing throughput and vector geometric dimensions.

### 4. Internal Batching with Guaranteed 1-to-1 Order Preservation
- **Location**: [`packages/genai-core/src/genai_core/services/embedding_service.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/services/embedding_service.py)
- **Decision**: Callers can supply arbitrary collections of texts (e.g. 1,000 document chunks) in a single `embed()` call. `EmbeddingService` slices the input into internal sub-batches of `batch_size`, invokes the provider, and concatenates the resulting vectors into `EmbeddingResponse.embeddings` in exact sequential 1-to-1 input order.
- **Alternatives Considered**: Forcing client applications to implement their own chunking and batch loops. Rejected because internal batching prevents memory exhaustion and standardizes slicing across all consumers.
- **Impact**: Prevents OOM crashes while maintaining predictable input-to-output mapping.

### 5. Honest Token Usage & Input Validation
- **Locations**:
  - [`packages/genai-core/src/genai_core/models/embedding_model/embedding_response.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/models/embedding_model/embedding_response.py)
  - [`packages/genai-core/src/genai_core/services/embedding_service.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/services/embedding_service.py)
- **Decision**:
  - `response.usage` is explicitly `None` when token counts are not natively emitted by the embedding model (e.g., local sentence-transformers). Never fabricate or guess token usage.
  - Passing empty list `input=[]` or empty/whitespace strings raises `ValueError("Input cannot be empty.")`.
- **Alternatives Considered**: Estimating tokens via string length heuristics (e.g., `len(text) // 4`). Rejected because heuristic token estimation is inaccurate and misleading for downstream billing or quota systems.
- **Impact**: Accurate metadata contracts and fail-fast validation on degenerate inputs.

### 6. Strictly Asynchronous Non-Blocking Execution
- **Locations**:
  - [`packages/genai-core/src/genai_core/embedding/local_provider.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/embedding/local_provider.py)
  - [`packages/genai-core/src/genai_core/services/embedding_service.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/services/embedding_service.py)
- **Decision**: All embedding service endpoints (`embed`, `embed_query`, `embed_documents`) are async coroutines. Local PyTorch inference runs via `asyncio.to_thread`.
- **Alternatives Considered**: Providing synchronous methods or running `model.encode()` directly on the main event loop thread. Rejected because CPU-bound PyTorch model execution blocks concurrent HTTP requests and async task scheduling.
- **Impact**: Consuming async applications (FastAPI, background workers) remain non-blocking during heavy vector embedding operations.

---

## 2026-10-17: Embedding Service Stage 3 — Hosted Providers and Shared Resilience Subsystem

### 1. Hosted Embedding Adapters (OpenAI and Google Gemini)
- **Locations**:
  - [`packages/genai-core/src/genai_core/embedding/openai_provider.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/embedding/openai_provider.py)
  - [`packages/genai-core/src/genai_core/embedding/google_provider.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/embedding/google_provider.py)
  - [`packages/genai-core/src/genai_core/embedding/__init__.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/embedding/__init__.py)
- **Decision**: Implemented `OpenAIEmbeddingProvider` and `GoogleEmbeddingProvider` conforming to the unified `EmbeddingProvider` protocol.
  - `OpenAIEmbeddingProvider`: Targets `text-embedding-3-small` (default, 1536 native dims) with truncation support via the `dimensions` parameter. Sorts response objects by API `item.index` to guarantee exact 1-to-1 input sequence preservation. Exposes honest token usage (`prompt_tokens`, `total_tokens`) in `last_usage`.
  - `GoogleEmbeddingProvider`: Uses current `text-embedding-005` (768 native dims) via `google-genai` SDK with `output_dimensionality` configuration. Strictly prohibits retired `text-embedding-004`. Safely extracts values from `EmbedContentResponse` and applies L2 unit normalization.
- **Alternatives Considered**: Making callers use separate APIs or instantiate SDK clients directly. Rejected because `genai-core`'s mission is to offer a consistent, interchangeable public API (`EmbeddingService`) across all providers.
- **Impact**: Consuming applications switch between local inference, OpenAI, and Google Gemini embeddings with zero code modifications to client vectorization logic.

### 2. Package-Level Shared Resilience Subsystem (`genai_core.resilience`)
- **Locations**:
  - [`packages/genai-core/src/genai_core/resilience/rate_limiter.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/resilience/rate_limiter.py)
  - [`packages/genai-core/src/genai_core/resilience/retry_policy.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/resilience/retry_policy.py)
  - [`packages/genai-core/src/genai_core/resilience/__init__.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/resilience/__init__.py)
- **Decision**: Created reusable resilience primitives in `genai_core.resilience`:
  - `RateLimiter`: Dual async token buckets (`AsyncTokenBucket`) tracking requests (RPM) and estimated tokens (TPM). Coroutines wait non-blocking via `asyncio.sleep` without stalling the event loop. Scope is explicitly documented as **per-process in-memory**. Supports dynamic rate-limit pauses upon receiving HTTP `Retry-After` headers.
  - `RetryPolicy` and `execute_with_retry`: Configurable retries (`max_retries=3`), exponential backoff (`initial_backoff=0.5s`, `max_backoff=10s`), full randomized jitter, per-attempt timeouts, and overall bounded deadlines.
  - Selective retryability: Transient network errors and HTTP 408/429/5xx are retried; non-retryable errors (400, 401, 403, quota exhaustion, dimension errors) fail fast without retrying.
- **Alternatives Considered**: Embedding resilience logic directly inside `EmbeddingService`. Rejected because `LLMService` and future retrieval/agent services require the identical token bucket and retry behavior, and duplicating resilience logic across services would lead to divergence and maintenance overhead.
- **Impact**: Provides a clean, reusable foundation for service stability across the entire package.

### 3. Internal Pacing Token Estimation vs. Honest Response Usage
- **Locations**:
  - [`packages/genai-core/src/genai_core/resilience/rate_limiter.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/resilience/rate_limiter.py)
  - [`packages/genai-core/src/genai_core/services/embedding_service.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/services/embedding_service.py)
- **Decision**: Added `estimate_tokens` (~4 characters per token heuristic) strictly for pacing TPM consumption inside `RateLimiter.acquire()`. These heuristic counts are internal pacing estimates and are **never** returned as actual usage in `EmbeddingResponse.usage`. `EmbeddingResponse.usage` is `None` for local models and aggregated from actual provider SDK counts for hosted models.
- **Alternatives Considered**: Returning heuristic token counts in `EmbeddingResponse.usage` for local models. Rejected because fabricating metadata misinforms callers and violates the honest usage contract.
- **Impact**: Accurate rate-limit pacing without corrupting client-facing usage metadata.

### 4. Bounded Request Deadlines Across Multi-Batch Requests
- **Location**: [`packages/genai-core/src/genai_core/services/embedding_service.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/services/embedding_service.py)
- **Decision**: For multi-batch requests, an overall request deadline (`time.monotonic() + timeout_seconds`) is established before the batch loop. Each sub-batch dynamically inherits the remaining deadline rather than resetting the clock, raising `EmbeddingTimeoutError` if the overall deadline is exceeded.
- **Alternatives Considered**: Giving every individual batch a fresh full timeout window. Rejected because a request with 100 batches could hang for 50 minutes (100 * 30s) without triggering a timeout.
- **Impact**: Guaranteed bounded execution time for large embedding jobs.

---

## 2026-10-19: Retrieval Architecture & In-Memory Implementation (Stages 1 & 2)

### 1. Retrieval Layer Boundaries & Extraction from Module 09
- **Locations**:
  - [`packages/genai-core/src/genai_core/models/retrieval_model/`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/models/retrieval_model/)
  - [`packages/genai-core/src/genai_core/services/retrieval_service.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/services/retrieval_service.py)
- **Decision**: Analyzed Module 09's RAG implementation and extracted core, reusable retrieval responsibilities while intentionally excluding application-specific concerns:
  - Excluded: Application-level text splitters/chunkers, document ingestion pipelines, hardcoded tenant logic, database schemas, and prompt synthesis templates.
  - Extracted: Core data contracts (`DocumentChunk`, `RetrievalQuery`, `RetrievalResult`, `RetrievalResponse`), vector store abstraction protocol, semantic search orchestration, metadata filtering, score thresholding, and ranking.
- **Alternatives Considered**: Copying Module 09's full RAG pipeline (including document chunkers, tenant parsers, and prompt templates) into `genai-core`. Rejected because ingestion and chunking vary widely across applications, whereas retrieval query execution and vector-store querying are universal primitives.
- **Impact**: Keeps `genai-core` clean and reusable across diverse future modules without imposing rigid document ingestion or database schemas.

### 2. Protocol-Driven Vector Store Architecture (`VectorStoreAdapter`)
- **Locations**:
  - [`packages/genai-core/src/genai_core/retrieval/protocol.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/retrieval/protocol.py)
  - [`packages/genai-core/src/genai_core/retrieval/in_memory_store.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/retrieval/in_memory_store.py)
- **Decision**: Defined `VectorStoreAdapter` as a `@runtime_checkable` `Protocol` with:
  - `dimensions: int`: Required geometric dimension matching.
  - `search(query_vector, top_k, score_threshold, filters) -> list[RetrievalResult]`: Async semantic search.
  - `upsert(chunks: list[DocumentChunk]) -> int`: Async vector indexing.
  - `delete(chunk_ids: list[str]) -> int`: Async vector deletion.
- **Alternatives Considered**: Using an abstract base class (`ABC`) or coupling `RetrievalService` directly to PGVector. Rejected because protocols provide duck-typing flexibility for third-party stores (Chroma, Qdrant, Pinecone) without forcing base class inheritance or heavy database client dependencies.
- **Impact**: Enables complete vector store interchangeability. `RetrievalService` consumes any conforming adapter.

### 3. Forward-Compatible Contract with Fail-Fast Search Mode Validation
- **Locations**:
  - [`packages/genai-core/src/genai_core/models/retrieval_model/retrieval_query.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/models/retrieval_model/retrieval_query.py)
  - [`packages/genai-core/src/genai_core/services/retrieval_service.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/services/retrieval_service.py)
  - [`packages/genai-core/src/genai_core/exceptions/retrieval_exceptions.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/exceptions/retrieval_exceptions.py)
- **Decision**: Added `search_mode: Literal["dense", "sparse", "hybrid"] = "dense"` to `RetrievalQuery`. In Phase 3 (Stages 1 & 2), only `"dense"` search is implemented. Requesting `"sparse"` or `"hybrid"` search immediately raises `UnsupportedSearchModeError`.
- **Alternatives Considered**: Faking hybrid search by falling back to dense search, or omitting `search_mode` until hybrid search is implemented. Rejected because silently ignoring requested search modes creates deceptive behavior for callers, while omitting the field would require breaking contract changes later.
- **Impact**: Clean forward-compatibility for sparse BM25 and hybrid reciprocal rank fusion (RRF) while strictly enforcing fail-fast behavior today.

### 4. Basic Store Mutation Operations (`upsert`, `delete`, `index_documents`)
- **Locations**:
  - [`packages/genai-core/src/genai_core/retrieval/protocol.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/retrieval/protocol.py)
  - [`packages/genai-core/src/genai_core/retrieval/in_memory_store.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/retrieval/in_memory_store.py)
  - [`packages/genai-core/src/genai_core/services/retrieval_service.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/services/retrieval_service.py)
- **Decision**: Vector stores implement `upsert()` and `delete()`. Furthermore, `RetrievalService` provides a convenience `index_documents(chunks: list[DocumentChunk])` method: if chunks are missing embeddings, `RetrievalService` automatically vectorizes them using its attached `EmbeddingService` before delegating to `store.upsert()`.
- **Alternatives Considered**: Pure read-only vector store interface. Rejected because testing, local development, and basic RAG workflows require a standard way to insert and remove chunks without bypassing the service abstraction.
- **Impact**: Provides end-to-end indexing and retrieval capability out of the box.

### 5. Strict Vector Dimension Validation
- **Locations**:
  - [`packages/genai-core/src/genai_core/services/retrieval_service.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/services/retrieval_service.py)
  - [`packages/genai-core/src/genai_core/retrieval/in_memory_store.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/retrieval/in_memory_store.py)
  - [`packages/genai-core/src/genai_core/exceptions/retrieval_exceptions.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/exceptions/retrieval_exceptions.py)
- **Decision**: Dimension compatibility is enforced at three levels:
  1. Service Initialization: If an `EmbeddingService` is provided, `retrieval_service.py` asserts `embedding_service.dimensions == store.dimensions`. A mismatch raises `IncompatibleEmbeddingError`.
  2. Direct Vector Queries: If a caller passes `query_vector` directly, its length is checked against `store.dimensions`. Mismatches raise `IncompatibleEmbeddingError`.
  3. Chunk Ingestion: `store.upsert()` verifies that all chunk embeddings match the store's configured dimension.
- **Alternatives Considered**: Allowing dimension mismatches to trigger math exceptions (e.g. dot product shape errors) during similarity calculation. Rejected because late mathematical errors obscure the root configuration defect.
- **Impact**: Immediate, actionable errors when components are misconfigured.

### 6. Zero-Dependency `InMemoryVectorStore`
- **Location**: [`packages/genai-core/src/genai_core/retrieval/in_memory_store.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/retrieval/in_memory_store.py)
- **Decision**: Implemented `InMemoryVectorStore` using pure Python standard library (`math`, `typing`). Calculates cosine similarity with floating-point tolerance, supports exact dictionary subset metadata filtering, filters results below `score_threshold`, and sorts by descending similarity score.
- **Alternatives Considered**: Using ChromaDB or FAISS for in-memory testing. Rejected to prevent adding heavy native C++ dependencies or third-party packages to base `genai-core`.
- **Impact**: Fast, lightweight, zero-dependency reference implementation ideal for unit testing, CI pipelines, and small in-memory retrieval tasks.

### 7. Resilience, Timeout Translation, and Observability Integration
- **Locations**:
  - [`packages/genai-core/src/genai_core/services/retrieval_service.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/services/retrieval_service.py)
  - [`packages/genai-core/src/genai_core/exceptions/retrieval_exceptions.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/exceptions/retrieval_exceptions.py)
  - [`packages/genai-core/src/genai_core/resilience/retry_policy.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/resilience/retry_policy.py)
- **Decision**:
  - In `retry_policy.py`, classified `IncompatibleEmbeddingError` and `UnsupportedSearchModeError` as non-retryable client errors.
  - Wrapped search operations with `execute_with_retry` using configured timeout (`retrieval_timeout_seconds = 30.0`).
  - Caught `EmbeddingTimeoutError` and `TimeoutError`, cleanly translating them to `RetrievalTimeoutError`.
  - Emitted structured observability events (`RETRIEVAL_START`, `RETRIEVAL_SUCCESS`, `RETRIEVAL_FAILED`, `RETRIEVAL_INDEX_START`, `RETRIEVAL_INDEX_SUCCESS`, `RETRIEVAL_DELETE_SUCCESS`) via `safe_log`. Text content is truncated to 100 characters in logs to avoid log bloat and privacy leaks.
- **Impact**: Robust operational behavior with comprehensive tracing and zero sensitive data leaks.

---

## 2026-10-19: PostgreSQL + pgvector Adapter Architecture (Stage 3 & Stage 4)

### 1. Injected Connection Pattern & Connection Lifecycle
- **Locations**:
  - [`packages/genai-core/src/genai_core/retrieval/pgvector_store.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/retrieval/pgvector_store.py)
  - [`packages/genai-core/src/genai_core/retrieval/protocol.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/retrieval/protocol.py)
- **Decision**: Analyzed Module 09's database implementation (`retrive_db.py`, `upsert_db.py`, `index_db.py`), which called `psycopg.AsyncConnection.connect()` on every operation. In `genai-core`, eliminated ad-hoc connection creation in favor of dependency-injected connections:
  - Supports injected connection pool (`psycopg_pool.AsyncConnectionPool` with `async with pool.connection() as conn:`), persistent connection, or connection factory (`Callable[[], Any]`).
  - Zero connection leaks: connections are acquired from the pool and released automatically.
- **Alternatives Considered**: Creating a new connection per query or embedding connection string parsing and pool configuration directly inside `PGVectorStoreAdapter`. Rejected because managing pools inside the store couples the library to specific connection configuration and prevents callers from sharing connection pools across multiple services.
- **Impact**: Optimal performance, pooling reuse, and seamless integration with FastAPI, background workers, or unit test mocks.

### 2. SQL Safety & Complete Query Parameterization
- **Location**: [`packages/genai-core/src/genai_core/retrieval/pgvector_store.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/retrieval/pgvector_store.py)
- **Decision**:
  - Enforced whitelist regex validation (`^[a-zA-Z_][a-zA-Z0-9_]*$`) on all table and column names (`table_name`, `embedding_column`, `id_column`, `text_column`, `metadata_column`, `tenant_id_column`, `conflict_target`). Invalid identifiers raise `ValueError` at initialization.
  - All query values (`query_vector`, filters, `score_threshold`, `top_k`, `chunk_ids`) are strictly passed as positional `%s` query parameters, completely eliminating SQL injection vulnerabilities.
- **Alternatives Considered**: Relying on caller trust for table/column names. Rejected because dynamic SQL generation must be defensively guarded against malicious or malformed identifier strings.
- **Impact**: Verified SQL safety against identifier injection and query value manipulation.

### 3. Optional Dependency Isolation (`postgres` Extra)
- **Locations**:
  - [`packages/genai-core/pyproject.toml`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/pyproject.toml)
  - [`packages/genai-core/src/genai_core/retrieval/pgvector_store.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/retrieval/pgvector_store.py)
  - [`packages/genai-core/src/genai_core/exceptions/embedding_exceptions.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/exceptions/embedding_exceptions.py)
- **Decision**: Declared PostgreSQL dependencies as an optional package extra: `[project.optional-dependencies] postgres = ["psycopg[binary]>=3.1.0", "pgvector>=0.2.0"]`. Base `genai-core` remains lightweight with zero database dependencies. Attempting to use the adapter without installed dependencies raises `MissingExtraDependencyError(extra="postgres", package="psycopg[binary] and pgvector")`.
- **Alternatives Considered**: Making `psycopg` and `pgvector` mandatory dependencies of `genai-core`. Rejected because consumers using hosted LLMs, embeddings, or in-memory retrieval should not be forced to install C-extensions and database drivers.
- **Impact**: Keeps the base package install fast and lightweight while offering plug-and-play PostgreSQL vector capabilities when the extra is installed.

### 4. JSONB Metadata Containment Filtering
- **Location**: [`packages/genai-core/src/genai_core/retrieval/pgvector_store.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/retrieval/pgvector_store.py)
- **Decision**: Implemented metadata filtering via PostgreSQL's JSONB containment operator (`metadata @> %s::jsonb`). Supports arbitrary dictionary subset matching identical to `InMemoryVectorStore`. Added optional `tenant_id_column` support for dedicated indexed column filtering (`WHERE tenant_id = %s`).
- **Alternatives Considered**: Creating dynamic SQL equality expressions for each metadata key (e.g. `metadata->>'key' = %s`). Rejected because JSONB containment (`@>`) leverages PostgreSQL GIN/jsonb_path_ops indexes and matches Python dict subset semantics cleanly.
- **Impact**: Consistent metadata filtering behavior across both in-memory and PostgreSQL backends.

### 5. Upsert & Delete Mechanics
- **Location**: [`packages/genai-core/src/genai_core/retrieval/pgvector_store.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/retrieval/pgvector_store.py)
- **Decision**:
  - Upsert: Parameterized batch `INSERT ... ON CONFLICT ({conflict_target}) DO UPDATE SET ...` updating content, vector, and metadata. Missing embeddings raise `ValueError`, and dimension mismatches raise `IncompatibleEmbeddingError`.
  - Delete: Parameterized `DELETE FROM {table_name} WHERE {id_col} = ANY(%s)`.
- **Alternatives Considered**: Separate insert-then-update or select-before-insert queries. Rejected because PostgreSQL `ON CONFLICT` is atomic and avoids race conditions.
- **Impact**: Safe, idempotent document indexing and deletion.

### 6. Fault Classification & Resilience Integration
- **Locations**:
  - [`packages/genai-core/src/genai_core/retrieval/pgvector_store.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/retrieval/pgvector_store.py)
  - [`packages/genai-core/src/genai_core/exceptions/retrieval_exceptions.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/exceptions/retrieval_exceptions.py)
  - [`packages/genai-core/src/genai_core/resilience/retry_policy.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/resilience/retry_policy.py)
- **Decision**: Added `retryable: bool = False` to `VectorStoreError`. `PGVectorStoreAdapter` detects transient operational errors (connection reset, broken pipe, connection closed, timeouts, deadlocks) and flags `VectorStoreError(retryable=True)`, allowing the shared `execute_with_retry` engine to automatically backoff and retry. Permanent errors (syntax errors, relation missing) fail fast.
---

## 2026-10-19: Test Suite Restructuring into Functional Subdirectories

### 1. Functional Categorization of Unit and Integration Tests
- **Locations**:
  - Unit tests: [`packages/genai-core/tests/unit/`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/unit/)
    - `embedding/`: `test_embedding_service.py`, `test_embedding_resilience.py`, `test_google_embedding_provider.py`, `test_openai_embedding_provider.py`
    - `retrieval/`: `test_retrieval_service.py`, `test_in_memory_store.py`, `test_pgvector_store.py`
    - `llm/`: `test_llm_service.py`, `test_anthropic_provider.py`, `test_google_provider.py`, `test_openai_provider.py`, `test_protocol.py`, `test_tool_spec.py`
    - `observability/`: `test_observability.py`
  - Integration tests: [`packages/genai-core/tests/integration/`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/integration/)
    - `llm/`: `test_anthropic_provider_integration.py`, `test_google_provider_integration.py`, `test_openai_provider_integration.py`
    - `observability/`: `test_observability_integration.py`
- **Decision**: Restructured flat test directories into functional subdirectories reflecting `module-09-rag`'s organization pattern and domain boundaries (`llm`, `embedding`, `retrieval`, `observability`). Added `__init__.py` markers across all test packages to ensure isolated pytest namespaces and type checking.
- **Alternatives Considered**: Keeping a single flat directory for all unit tests and another for integration tests. Rejected as the test suite grew to 18+ test files across multiple distinct services, making test file discovery and selective test execution clumsy.
- **Impact**: Clean domain isolation, clear separation of concerns, faster test file discovery, and 100% preservation of test assertions and coverage.

---

## 2026-10-19: Hybrid Retrieval, Full-Text Keyword Search & Cross-Encoder Re-ranking

### 1. Unified Search Mode Dispatch (`dense`, `sparse`, `hybrid`)
- **Locations**:
  - [`packages/genai-core/src/genai_core/models/retrieval_model/retrieval_query.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/models/retrieval_model/retrieval_query.py)
  - [`packages/genai-core/src/genai_core/services/retrieval_service.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/services/retrieval_service.py)
- **Decision**: Expanded `search_mode` in `RetrievalQuery` to fully support `"dense"`, `"sparse"`, and `"hybrid"`:
  - `"dense"`: Evaluates semantic cosine similarity using embedding vectors.
  - `"sparse"`: Bypasses embedding generation completely and delegates directly to full-text keyword search (`search_keyword`), saving API token costs and execution time.
  - `"hybrid"`: Concurrently executes vector search and keyword search via `asyncio.gather`.
- **Alternatives Considered**: Creating separate service classes (e.g. `KeywordRetrievalService`, `HybridRetrievalService`). Rejected because callers benefit from a single, unified service entry point where search mode is declared via query parameters.
- **Impact**: Clean, unified contract with zero redundant service wrappers.

### 2. Standalone Min-Max Normalization & Weighted Fusion Engine
- **Location**: [`packages/genai-core/src/genai_core/retrieval/fusion.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/retrieval/fusion.py)
- **Decision**: Implemented standalone mathematical functions for score normalization and result fusion:
  - `normalize_min_max`: Scaled scores linearly to `[0.0, 1.0]`. Identical scores or singletons normalize to `1.0`.
  - `weighted_score_fusion`: Blends normalized vector and keyword candidate lists using configurable weights (default `0.6` vector, `0.4` keyword). Candidates present in only one channel retain that channel's weighted score.
- **Alternatives Considered**: Embedding fusion math directly inside `RetrievalService.search`. Rejected because isolating fusion into pure functions ensures deterministic, standalone unit testability without mocking services or stores.
- **Impact**: 100% testable fusion logic matching Module 09's proven score semantics.

### 3. PostgreSQL Full-Text Search with Dynamic Fallback
- **Location**: [`packages/genai-core/src/genai_core/retrieval/pgvector_store.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/retrieval/pgvector_store.py)
- **Decision**: Added `search_keyword` using PostgreSQL's `websearch_to_tsquery` and `ts_rank_cd`:
  - When `search_vector_column` is configured, searches that indexed `tsvector` column directly.
  - When `search_vector_column` is `None`, dynamically computes `to_tsvector(%s, {text_column}) @@ websearch_to_tsquery(%s, %s)`, allowing instant full-text search on any existing table without database migrations.
  - Enforced strict regex identifier whitelisting (`^[a-zA-Z_][a-zA-Z0-9_]*$`) on column and language identifiers. All query values are passed strictly as `%s` positional parameters.
- **Alternatives Considered**: Requiring a pre-existing `tsvector` column on all tables. Rejected because forcing schema migrations on client tables reduces library reusability across existing databases.
- **Impact**: Flexible, secure full-text search across both existing and dedicated PostgreSQL schemas.

### 4. Modular Cross-Encoder Re-ranking with Non-Wasting Fallback
- **Locations**:
  - [`packages/genai-core/src/genai_core/retrieval/protocol.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/retrieval/protocol.py)
  - [`packages/genai-core/src/genai_core/retrieval/cross_encoder_reranker.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/retrieval/cross_encoder_reranker.py)
  - [`packages/genai-core/src/genai_core/services/retrieval_service.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/services/retrieval_service.py)
- **Decision**:
  - Defined `@runtime_checkable` `Reranker` protocol.
  - Implemented `CrossEncoderReranker` using `sentence-transformers.CrossEncoder` (default `cross-encoder/ms-marco-MiniLM-L6-v2`) with lazy loading and `asyncio.to_thread` for non-blocking execution.
  - Re-ranking is strictly optional (disabled by default) and controlled via `Settings`, `RetrievalService(enable_reranking=...)`, or `RetrievalQuery(rerank=True)`.
  - Preserved initial retrieval scores in `result.metadata["retrieval_score"]` for complete auditability.
  - Supported caller-configurable fallback (`on_rerank_unavailable`): when set to `"warn_and_continue"` (default), if re-ranking dependencies are missing or unconfigured, the service logs `RETRIEVAL_RERANK_SKIPPED`, returns un-reranked candidates so retrieval computation is not wasted, sets `response.reranked = False`, and populates `response.rerank_message`.
- **Alternatives Considered**: Automatically executing re-ranking on every query or failing silently when re-ranking was requested but unavailable. Rejected because re-ranking is computationally expensive and silent failure obscures missing capabilities.
- **Impact**: Production-grade re-ranking under caller control with zero wasted work on fallback.

---

## 2026-10-19: Live Integration Tests for Retrieval and Embedding Subsystems

### 1. Dedicated Integration Test Structure
- **Locations**:
  - [`packages/genai-core/tests/integration/embedding/`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/integration/embedding/)
  - [`packages/genai-core/tests/integration/retrieval/`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/tests/integration/retrieval/)
- **Decision**: Implemented dedicated live integration test suites:
  - `tests/integration/embedding/test_local_embedding_integration.py`: Validates real `LocalEmbeddingProvider` (`all-MiniLM-L6-v2`) inference, vector dimensionality (384), L2 unit normalization ($\approx 1.0$), and semantic cosine distances.
  - `tests/integration/embedding/test_google_embedding_integration.py` & `test_openai_embedding_integration.py`: Validates live cloud embedding requests with graceful skip handling for missing API keys, rate limits, or unavailable endpoints.
  - `tests/integration/retrieval/test_cross_encoder_reranker_integration.py`: Validates real neural re-ranking using `cross-encoder/ms-marco-MiniLM-L6-v2`, score ranking, and retrieval score tracking.
  - `tests/integration/retrieval/test_pgvector_store_integration.py`: Validates live PostgreSQL vector search, full-text keyword search (`ts_rank_cd`), metadata JSONB filtering, and deletion against PostgreSQL.
  - `tests/integration/retrieval/test_hybrid_retrieval_integration.py`: End-to-end integration across `LocalEmbeddingProvider`, `PGVectorStoreAdapter`, and `CrossEncoderReranker` orchestrated via `RetrievalService`.
- **Alternatives Considered**: Mocking all databases and model inferences across all tests. Rejected because live integration tests are essential for detecting SQL compatibility, type adaptation, and third-party driver regressions.
- **Impact**: Complete end-to-end confidence under realistic runtime environments with zero regressions on fast unit test suites.

### 2. Seamless Vector Parameter Adaptation (`_to_vector_param`)
- **Location**: [`packages/genai-core/src/genai_core/retrieval/pgvector_store.py`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/retrieval/pgvector_store.py)
- **Decision**: Added `_to_vector_param(vec: list[float]) -> Any` helper which wraps float lists in a `_VectorParam` subclass of `pgvector.Vector`. It provides standard `__eq__` sequence equality so unit test mock assertions pass without modifications, while informing `psycopg` to dump vector types over the wire. This avoids overriding the global `list` dumper on database connections, which previously broke `DELETE ... ANY(%s)` with `list[str]`.
- **Alternatives Considered**: Modifying SQL queries to use explicit `%s::vector` casts everywhere or registering dumpers globally on `conn.adapters`. Global dumpers broke other list parameters (such as string lists), and SQL casts break databases where pgvector type is registered under different search paths.
- **Impact**: Fully robust PostgreSQL pgvector operations for search, upsert, and delete without side effects on non-vector list parameters.

