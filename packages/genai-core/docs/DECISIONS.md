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



