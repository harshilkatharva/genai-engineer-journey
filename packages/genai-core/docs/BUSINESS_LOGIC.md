# BUSINESS_LOGIC.md — `genai-core`

> **Document type:** Functional responsibilities, behavioral rules, component boundaries, and interaction contracts  
> **Status:** Living specification; future capabilities are explicitly marked as planned  
> **Intended location in the repository:** `packages/genai-core/docs/BUSINESS_LOGIC.md`

## 1. Purpose and interpretation

This document describes how `genai-core` is expected to behave as a shared Python package. It defines responsibilities and rules at the component boundary rather than locking the implementation to a particular class layout, provider SDK, or framework.

The package begins with **LLM service integration**. Embeddings, retrieval, memory, tool/function-calling primitives, and other shared capabilities may follow as Modules 17–20 demonstrate a need. A planned behavior in this document is not proof that the feature has already been implemented.

Use these status labels when reading or updating this document:

- **Initial:** Required for the first usable LLM-service milestone or package foundation.
- **Planned:** A future capability whose contract should be refined when a consuming use case appears.
- **Optional:** Useful when needed, but not required for every consumer or installation.
- **Application-owned:** Intentionally remains the responsibility of the mini-project or application.

## 2. Core business rule

**A consuming application requests a capability through a clear public contract; `genai-core` validates the request, applies the shared behavior, delegates provider- or storage-specific work to the appropriate integration boundary, normalizes the result where meaningful, and returns a predictable result or actionable error.**

The core must not silently take ownership of the consuming application's domain workflow, authorization decisions, or business policy merely because those decisions involve a model.

## 3. Package and component boundaries

### 3.1 Consumer boundary

Mini-projects and applications consume documented public APIs. They supply required inputs and configuration, compose capabilities, decide application-specific workflows, and handle their own user-facing behavior.

### 3.2 Shared-service boundary

A shared service implements common behavior and enforces its contract. For example, an LLM service may validate a generation request, select the configured integration, invoke it, and return a normalized response.

### 3.3 Integration boundary

Provider, vector-store, memory-store, or framework adapters translate between a shared contract and an external library/service. SDK-specific objects should not leak into the general contract unless exposing them is deliberate and documented.

### 3.4 External-service boundary

External APIs and stores can fail, have provider-specific limitations, enforce quotas, and change independently. `genai-core` should surface these facts through useful errors and metadata rather than presenting external operations as infallible local functions.

### 3.5 Dependency-direction rules

- Applications may depend on the public API of `genai-core`.
- Core code must not depend on a consuming application's package or module.
- Core components must not import one another in circular ways.
- Imports must not make network calls, create persistent connections, or require real credentials as a side effect.
- Optional integrations should be loaded or installed in a way that does not burden consumers that do not use them.

## 4. Rules that apply to all capabilities

### 4.1 Validate before performing external work

Validate required inputs, supported options, configuration, and capability requirements before issuing a network request or mutating persistent state whenever possible. Invalid input should result in a clear validation/configuration error, not a confusing provider failure later.

### 4.2 Keep configuration explicit

A consumer should be able to determine which provider, model, endpoint, and relevant options are active. Environment variables or secret stores may provide values, but the source and precedence of configuration must be documented. Avoid hidden process-wide mutable configuration.

If multiple configuration sources are supported, define an explicit precedence rule. Explicit per-call settings may override defaults only where that override is part of the contract. Conflicting or invalid settings should fail clearly instead of being silently resolved in a surprising way.

### 4.3 Treat credentials and sensitive data carefully

- Never hard-code credentials in source code or committed example files.
- Never include credentials in exceptions, logs, debug dumps, or test snapshots.
- Do not log full prompts, retrieved documents, tool arguments, or memory contents by default.
- If diagnostics need request data, make that behavior opt-in, document it, and support redaction where practical.
- Do not send a user's data to another provider or service unless the configured capability requires it and the consumer has chosen that integration.

### 4.4 Make external behavior predictable

External calls should support sensible timeout behavior. Retries should be bounded and configurable where appropriate. Only transient failures should normally be retried automatically. Authentication failures, malformed requests, validation errors, and unsupported capabilities should not be retried as though they were temporary network failures.

Do not automatically repeat a potentially side-effecting tool execution unless retry safety or idempotency is established.

### 4.5 Preserve useful failure information

Errors should identify the broad failure category and give enough context to help a consumer recover. They should not expose secrets or unnecessarily leak raw provider payloads. When an error is translated from an SDK exception, preserve the original exception as a cause when practical.

### 4.6 Do not silently ignore options

If a requested option or capability is unsupported by the selected integration, `genai-core` must either reject it clearly or follow an explicitly documented fallback. It must not silently discard the option and report success.

### 4.7 Keep state and side effects visible

A capability that makes network calls, writes memory, mutates a store, or executes a tool should make that behavior clear. Avoid hidden persistent state or implicit global sessions. Reusable services should be testable with injected fakes/adapters where appropriate.

### 4.8 Separate provider-independent contracts from provider-specific features

Normalize common behavior only where the meaning is genuinely shared. Preserve meaningful differences—such as supported models, structured-output modes, token limits, streaming support, tool-call semantics, or usage metadata—through explicit capability metadata or documented provider-specific configuration.

## 5. Initial capability: LLM service integration

This is the first service to implement. The rules below describe the initial intended behavior; exact public class and method names should be decided in the package API rather than assumed from the wording of this document.

### 5.1 Responsibility

The LLM service provides a consistent way for a consumer to request model generation without embedding provider SDK setup, request translation, response parsing, and error handling in every project.

It is responsible for:

- accepting a documented generation request;
- validating required input and options;
- using the configured provider/model integration;
- converting the shared request into the provider's request format;
- invoking the configured external service;
- translating the result into the documented common response shape;
- translating integration failures into useful core errors;
- exposing relevant metadata and capability limitations without leaking credentials.

It is not responsible for owning the full application workflow, determining whether generated content is appropriate for a particular business domain, or automatically invoking arbitrary application tools.

### 5.2 Request expectations

A generation request should have a clear, documented representation. Depending on the use case, it may include:

- user content or a sequence of role-labelled messages;
- a selected model or a documented configured default;
- generation parameters that the shared contract supports;
- an optional timeout or request metadata if justified by the API;
- optional structured-output or tool-calling settings only when those capabilities are supported by the selected integration.

The public contract must define whether message-based requests or simple text prompts are primary. If both are supported, define how simple text is converted into the message representation. Valid roles and required content must be checked before a provider call.

Do not add every possible provider parameter to the common contract. Provider-specific parameters should be explicit and namespaced (or supplied through a documented adapter-specific extension) so that they cannot be confused with portable settings.

### 5.3 Response expectations

A successful response should provide a predictable representation of the model's result. Where available and useful, the normalized response may include:

- generated text or structured content;
- finish/termination reason;
- model and provider identifiers;
- token-usage information;
- request/response identifiers or timing metadata where safe and available;
- tool-call requests or other structured items when the corresponding capability is enabled.

Optional provider metadata must be marked as optional. The response contract must not imply that every provider reports every field. If the raw provider response is made accessible for advanced use, it should be clearly marked provider-specific and must not be required for normal application behavior.

### 5.4 Provider selection and capability rules

- The selected provider and model must be explicit or come from a documented default configuration.
- The service must fail clearly when configuration is missing, contradictory, or unsupported.
- Provider-specific SDK initialization and response conversion belong inside the integration boundary.
- A provider adapter must not claim support for features it does not actually implement.
- Differences that cannot be normalized safely must remain visible in capability information or documented provider-specific behavior.
- Adding a provider should not require changing unrelated consumer workflows when the requested capability fits the shared contract.

### 5.5 Error categories

The initial service should distinguish, at least conceptually, these failure categories:

| Category | Meaning | Expected handling |
|---|---|---|
| Validation error | Request fields or options are invalid. | Correct the input; do not retry unchanged. |
| Configuration error | Required provider/model configuration is missing or invalid. | Fix configuration; do not retry unchanged. |
| Authentication/authorization error | Credentials or permissions are invalid/insufficient. | Correct credentials/permissions; do not retry unchanged. |
| Rate-limit/quota error | Provider is throttling or quota is exhausted. | Expose the category and any safe retry guidance; retry only under bounded policy. |
| Timeout/network error | The external call did not complete normally. | Apply bounded retry only where configured and appropriate. |
| Provider/service error | The provider reports a service-side failure. | Preserve actionable context without leaking sensitive payloads. |
| Unsupported capability | The provider cannot perform a requested operation. | Choose a supported path or change the request. |
| Invalid provider response | The integration cannot parse or validate the returned data. | Report an integration/protocol failure; do not return a fabricated success. |

Specific exception names and inheritance should be selected during implementation. The semantic categories should remain clear even if the internal exception hierarchy evolves.

### 5.6 Retry and idempotency rules

Retries must have a limit and must not form unbounded loops. Backoff and provider `Retry-After` handling may be supported where useful. The service must not retry invalid requests or authentication failures by default. A timeout does not always prove that the provider did not process the request; duplicate cost or work is possible. Document this uncertainty when relevant.

### 5.7 Testing rules

- Unit tests must not require real API keys or internet access.
- Request validation and response normalization must be testable with deterministic fakes.
- Each provider adapter should have tests for translation and error mapping.
- Live provider tests should be separate, opt-in integration tests.
- Tests must verify that secrets and sensitive values are not included in expected error/log output.
- At least one small consumer example should demonstrate that application code uses the public contract rather than provider internals.

### 5.8 Not automatically required in the first milestone

Streaming, automatic tool execution, automatic retries beyond the basic safe policy, caching, routing across many providers, prompt management systems, tracing platforms, and complex fallback chains are **not required by default**. Add them only when a module use case justifies their cost and the behavior can be specified and tested clearly.

## 6. Implemented capability: embeddings

Embeddings are a separate capability from text generation, even when provided by the same vendor. The service is implemented via [`EmbeddingService`](file:///home/harshilk/genai-engineer-journey/packages/genai-core/src/genai_core/services/embedding_service.py) with the following confirmed business rules:

### 6.1 Provider & Model Defaults
- Default provider is `local` (using SentenceTransformers).
- Default model is `sentence-transformers/all-MiniLM-L6-v2` (`all-MiniLM-L6-v2`).
- Default native dimension is `384`.
- Default internal batch size is `32`.
- `sentence-transformers` is an optional package extra (`genai-core[local]`), keeping base `genai-core` installation lightweight. If absent, attempting local embedding raises `MissingExtraDependencyError` with instructions to install `genai-core[local]`.

### 6.2 Strict Dimension Validation
- Dimensions are configured and validated at service initialization time (`EmbeddingService(dimensions=...)`).
- Unsupported dimensions are strictly rejected whenever model capabilities are known or inspectable. For example, `all-MiniLM-L6-v2` strictly rejects `dimensions=768` by raising `EmbeddingDimensionError`.
- Vector dimensions in returned batches are verified against configured dimensions; mismatches raise `EmbeddingDimensionError`.

### 6.3 Input Validation & Empty Input Policy
- `embed(input=[])` raises `ValueError("Input cannot be empty.")`.
- Empty strings (`""`) or whitespace-only strings (`"   "`) passed as input raise `ValueError("Input text cannot be empty or whitespace only.")`.
- Any item within an input list that is empty or whitespace-only raises `ValueError` identifying the invalid index.
- Non-string / non-list inputs raise `TypeError`.

### 6.4 Batch Size & Internal Batching
- Batch size is configurable at the service level and overridable per request via `EmbeddingRequest(batch_size=...)`.
- Batch size is completely distinct from vector dimensions.
- Batch size `<= 0` raises `ValueError("batch_size must be greater than 0.")`.
- Large inputs (e.g. 1,000 texts) are sliced internally into sub-batches of `batch_size`.
- **Strict 1-to-1 Order Preservation**: Output vectors in `EmbeddingResponse.embeddings` strictly match the index order of input texts.

### 6.5 Token Usage & Metadata Semantics
- When token usage metrics are unavailable (such as local model embeddings), `response.usage` is explicitly `None`.
- Token counts are never fabricated, estimated, or guessed.
- `response.latency_ms` records total wall-clock duration in milliseconds.

### 6.6 Asynchronous Non-Blocking Execution
- All embedding endpoints (`embed`, `embed_query`, `embed_documents`) are strictly asynchronous coroutines.
- CPU-bound local model inference runs in a worker thread via `asyncio.to_thread`, keeping the Python asyncio event loop responsive.

### 6.7 Hosted Provider Rules (OpenAI & Google)
- **OpenAI (`OpenAIEmbeddingProvider`)**:
  - Default model is `text-embedding-3-small` (1536 native dimensions). Also supports `text-embedding-3-large` (3072 native dimensions) and legacy `text-embedding-ada-002` (1536 fixed dimensions).
  - Supports variable `dimensions` parameter for `text-embedding-3-*` models down to any positive integer `<= max_dimensions`. Legacy `text-embedding-ada-002` strictly rejects non-1536 dimensions.
  - Guarantees 1-to-1 index-ordered outputs by explicitly sorting response objects on the `item.index` attribute returned by OpenAI API.
  - Aggregates actual token usage (`prompt_tokens`, `total_tokens`) reported in OpenAI API response into `EmbeddingResponse.usage`.
- **Google Gemini (`GoogleEmbeddingProvider`)**:
  - Uses current supported `text-embedding-005` model (768 native dimensions). Retired `text-embedding-004` is forbidden and rejected.
  - Passes `output_dimensionality=dimensions` via `EmbedContentConfig` when a dimension `<= 768` is configured.
  - Safely extracts values from `EmbedContentResponse` and computes L2 unit normalization.
  - Provider errors are caught and converted to `EmbeddingProviderError("google", ...)`.

### 6.8 Shared Resilience: Rate Limiting & Bounded Retries
- **In-Memory Asynchronous Rate Limiter (`RateLimiter`)**:
  - Enforces configurable RPM (requests per minute) and TPM (tokens per minute) using an async dual-token bucket (`AsyncTokenBucket`).
  - Thread/coroutine-safe via `asyncio.Lock` with non-blocking waits (`asyncio.sleep`).
  - Scope is strictly **per-process in-memory**. Does not require distributed locks or external Redis coordination in this phase.
  - **Internal Pacing Token Estimation**: Uses `estimate_tokens` (~4 characters per token heuristic) strictly for internal TPM bucket pacing. These estimates are never exposed as actual usage metadata in `EmbeddingResponse`.
  - **`Retry-After` Handling**: Inspects provider rate-limit responses and pauses dispatch dynamically when a `Retry-After` header is returned.
- **Retry Policy & Deadlines (`RetryPolicy` & `execute_with_retry`)**:
  - Default policy: `max_retries=3`, `initial_backoff=0.5s`, `max_backoff=10.0s`, with full randomized jitter to avoid synchronized retry storms.
  - **Selective Retryability**: Transient errors (network connection errors, HTTP 408, 429, 500, 502, 503, 504) are retried with exponential backoff.
  - **Fail-Fast Policy**: Authentication failures, invalid inputs, dimension errors, missing extras, and permanent quota/billing failures (`insufficient_quota`, `billing_not_active`, `account_deactivated`) immediately fail without retrying.
  - **Bounded Request Deadline**: Evaluates `time.monotonic()` against an overall deadline across all sub-batches. Times out with `EmbeddingTimeoutError` if exceeded.

### 6.9 Observability Integration
- Emits `EMBEDDING_REQUEST_START` and `EMBEDDING_REQUEST_SUCCESS` via `safe_log` with provider, model, dimensions, total inputs, and latency.
- Emits `EMBEDDING_RATE_LIMITED` when the rate limiter throttles and delays a batch.
- Emits `EMBEDDING_RETRY` on transient error retry attempts with attempt index, sleep delay, and error message.
- Emits `EMBEDDING_TIMEOUT` when the request deadline or attempt timeout expires.
- Provider errors emit `EMBEDDING_REQUEST_FAILED` with error details and raise `EmbeddingProviderError`.

**Status:** Completed (Stages 1, 2, and 3 verified by 98 unit tests and package pre-push checks).


## 7. Planned capability: retrieval and document abstractions

When multiple projects require shared retrieval behavior, the package may define contracts for documents/chunks, retrieval queries, filters, and results, along with adapters for selected backends.

Intended rules:

- A document/chunk should keep its content distinct from metadata.
- Stable IDs and source metadata should be retained where provided.
- Retrieval results should preserve useful source information and relevant score/rank data when the backend provides it.
- Optional metadata and scores must not be represented as guaranteed when a backend cannot supply them.
- Filters and retrieval options must either be honored or rejected clearly; they must not be silently dropped.
- Backend-specific features may remain accessible through explicit adapter-specific extensions.
- Retrieval should return evidence/results; deciding how to cite, synthesize, or present it is generally application/workflow behavior unless a shared component is deliberately defined for that purpose.

Do not force all vector databases or search services into an interface that hides important differences. Start with the smallest shared contract required by the first consumers.

**Status:** Planned; storage-specific implementations and dependency choices are intentionally deferred until requirements are known.

## 8. Planned capability: memory

Memory must be explicit about what is stored, how it is addressed, how it is retrieved, and how long it is retained.

When implemented, it should define:

- the unit being stored (for example, a message, summary, fact, or memory record);
- the scope or namespace (for example, conversation/session or user/application scope);
- the store interface and supported operations;
- retrieval/search behavior and relevant metadata;
- update, deletion, expiration, and retention expectations where supported;
- behavior when storage is unavailable or a record is missing;
- rules for isolating memory belonging to different sessions/users when the application uses those boundaries.

Rules:

- Do not silently share memory across unrelated users or sessions.
- Do not assume all memory is persistent, global, or automatically attached to every prompt.
- Do not treat retrieved memory as trusted instructions; consumers must apply their own trust and policy rules.
- Make writes and deletes explicit, and document any eventual-consistency or persistence behavior.
- Avoid storing sensitive content by default without a clear use case and appropriate safeguards.

**Status:** Planned; the exact memory model should be guided by the modules' actual use cases rather than chosen prematurely.

## 9. Planned capability: tool/function-calling primitives

Tool/function calling has two distinct parts: the model may **request** a tool call, and application code decides whether and how to **execute** it. A model-generated request is not authorization to run arbitrary code.

If shared tool/function-calling contracts are introduced, they should:

- represent tool name, description/schema, arguments, and result consistently;
- validate that a requested tool exists and that its arguments satisfy the declared schema;
- normalize provider-specific tool-call representations without erasing meaningful differences;
- distinguish malformed calls, unknown tools, validation failures, execution failures, and successful results;
- preserve correlation identifiers where needed to associate tool results with model requests;
- make timeout, retry, and side-effect concerns explicit.

The consuming application owns the allowlist of available tools, authorization checks, permission boundaries, approval requirements, and the decision to execute a requested tool. Core may provide reusable validation and dispatch primitives, but it must not execute arbitrary names, shell commands, code, or network operations just because model output requests them.

Automatic orchestration loops should be introduced only when clearly required and designed with bounded steps, error handling, and termination rules.

**Status:** Planned; implement only after a concrete cross-project use case validates the shared contract.

## 10. Framework and orchestration interoperability

A mini-project may use LangChain, LangGraph, LlamaIndex, custom Python logic, or another tool where appropriate. `genai-core` should not require all applications to adopt a single framework merely to reuse basic services.

When an integration adapter is justified:

- keep the integration optional when practical;
- make framework-specific behavior visible at the boundary;
- avoid making framework objects the required return type of a general-purpose core API unless that choice is deliberate;
- do not duplicate a mature framework feature without a clear reason;
- do not move a module's educational orchestration logic into core just to remove a few lines of code.

**Status:** Integration policy applies now; specific adapters are optional and demand-driven.

## 11. Shared configuration behavior

The package should establish common configuration conventions without imposing one configuration mechanism on every consumer.

Expected rules:

- Required settings must be validated before use.
- Defaults must be documented and safe.
- Precedence between explicit arguments, configuration objects, environment values, and defaults must be predictable if more than one source is supported.
- Secret values must be redacted from string representations, logs, and errors.
- Configuration must identify the provider and model (or equivalent service target) unambiguously.
- Provider-specific values should be separated from portable shared settings.
- Missing optional integrations should produce a helpful error when that integration is invoked, not a confusing import failure in unrelated code.
- Configuration should be injectable for tests and separate instances where that helps avoid global-state coupling.

Exact variable names, configuration classes, and file formats should be decided and documented during implementation.

## 12. Shared data contracts and public API

The package should distinguish supported public contracts from implementation details.

Rules:

- Public inputs and outputs should be documented and typed where appropriate.
- Contracts should describe required fields, optional fields, defaults, and failure behavior.
- Optional metadata must be clearly optional.
- Consumers should not need to import a private module path to use a supported service.
- Provider SDK objects must not leak into shared types accidentally.
- If a raw or provider-specific escape hatch is supported, document its compatibility limitations.
- Experimental interfaces should be marked as such and not presented as equally stable as supported public APIs.
- Public exports should have one documented route where practical; avoid duplicate aliases that create ambiguity.

The exact layout (`services`, `providers`, `models`, or another organization) is an implementation decision. The behavior contract should remain valid if internal files are reorganized.

## 13. Dependency and packaging rules

`genai-core` is a package consumed by projects inside the repository and potentially later by other projects. Packaging and dependency behavior must therefore be deliberate.

- Declare required dependencies separately from optional provider/framework integrations where the packaging setup supports it.
- Do not require every supported SDK or framework for a consumer that uses only one capability.
- Avoid undeclared reliance on the current working directory or manually modified Python import paths.
- Use a consistent package import name and document the intended installation/editable-install workflow for the repository.
- Test the package from a consumer context, not only from inside its own directory.
- Avoid circular imports and accidental dependencies on sibling mini-projects.
- Document supported Python/runtime versions and external dependency constraints when those decisions are established.

The repository's exact build backend and dependency manager should be preserved or deliberately decided in the implementation; this document does not assume one in advance.

## 14. Error behavior across components

All components should distinguish errors caused by the consumer's input/configuration from failures in external systems and failures inside core itself.

A useful conceptual taxonomy is:

- **Input/validation:** supplied data violates the component's contract.
- **Configuration:** required settings or credentials are missing or invalid.
- **Unsupported operation/capability:** the requested behavior is not implemented by this component or selected integration.
- **External service/store:** provider or persistence dependency failed.
- **Timeout/rate limit:** an external operation did not complete or was throttled.
- **Data/protocol:** external output is malformed or cannot satisfy the expected response contract.
- **Internal/unexpected:** a bug or unanticipated condition occurred inside the package.

Consumers should be able to catch the appropriate category without relying on raw SDK exception classes in the shared path. Do not convert every failure into an empty result, a generic success-shaped response, or a swallowed exception.

## 15. Observability and logging

Observability is implemented as a reusable, caller-controlled core capability (`genai_core.observability`).

Verified rules:

- **Caller-controlled destination:** Logs are emitted to destinations configured by the consuming application (`ObservabilityConfig.log_file` or console stdout). By default (`NullHandler`), zero files or side effects are created on disk.
- **Format selection:** Supports structured single-line JSON (`JsonFormatter`) and human-readable text (`TextFormatter`).
- **Security & Secret Scrubbing:** `RedactingFilter` automatically scrubs Google, OpenAI, Anthropic API keys, Bearer tokens, and credential dictionary keys (`api_key`, `secret`, `password`, `token`) from all log records, messages, and exception tracebacks. Token usage metrics (`input_tokens`, `output_tokens`) are preserved.
- **Request tracing:** Async request context is managed via `observability_context` and `get_request_id` without requiring intrusive function argument passing. Context lookups return `None` or auto-generate UUIDs gracefully without raising exceptions.
- **Fault isolation:** All internal service logging calls dispatch via `safe_log`. Handler or filesystem errors are caught and suppressed, ensuring logging failures never break primary business logic or LLM completions.

**Status:** Implemented and integrated with `LLMService` operations.

## 16. Testing and compatibility behavior

### 16.1 Testing layers

1. **Unit tests:** Validate pure business rules, validation, mapping, normalization, configuration, and error conversion without live services.
2. **Adapter tests:** Check how an integration translates inputs/outputs and maps errors. Use fakes or controlled fixtures where possible.
3. **Integration tests:** Exercise real provider/store behavior only when explicitly enabled and correctly configured.
4. **Consumer tests/examples:** Prove that a small mini-project can install/import and use the public API as documented.

The normal fast test suite should not fail because a developer lacks a provider key or internet access.

### 16.2 Compatibility rules

- Changes to a supported public contract require deliberate review.
- Prefer additive changes when possible.
- When removal or incompatible changes are necessary, document the reason, migration path, and affected consumers.
- Run relevant core tests and consumer-facing tests before considering a change complete.
- Do not promise backward compatibility for an explicitly experimental API, but clearly label it as experimental.

## 17. Component interaction patterns

### 17.1 Initial LLM request flow

```text
Mini-project
   |
   | creates a documented request + supplies configuration
   v
LLM service (public contract)
   |
   | validates input, options, and configuration
   v
Provider integration / adapter
   |
   | translates request and calls external SDK/API
   v
LLM provider
   |
   | response or failure
   v
Provider integration / adapter
   |
   | normalizes result or maps error
   v
LLM service
   |
   | returns shared response or actionable core error
   v
Mini-project
```

The application remains responsible for deciding what to do with the result—for example, displaying it, evaluating it, storing it, combining it with retrieved context, or continuing a workflow.

### 17.2 Future retrieval-augmented flow (illustrative)

```text
Application workflow
   |                         |
   v                         v
Embedding service        Retrieval component
   |                         ^
   | creates query vector    | queries configured store/index
   v                         |
Configured embedding      Retrieval adapter/store
   |
   +-------------------------+

Application workflow combines retrieved evidence with its prompt/request
   |
   v
LLM service
   |
   v
Application decides how to validate, cite, store, or present the result
```

This is an illustration of possible composition, not a requirement to place the entire RAG workflow inside `genai-core`. A shared retrieval utility can be core-owned while a project's RAG strategy remains application-owned.

### 17.3 Future memory-assisted flow (illustrative)

```text
Application identifies conversation/session scope
   |
   v
Memory interface <--> configured memory store/retriever
   |
   v
Application chooses relevant memory and forms a request
   |
   v
LLM service
   |
   v
Application chooses whether to save a new memory
```

Memory retrieval, prompt assembly, write policy, and retention should be explicit. The core must not silently attach every stored record to every model request.

## 18. Ownership decision table

| Concern | Default owner | Reason |
|---|---|---|
| Provider request translation | Core adapter | Repeated integration detail that should not be copied into every consumer. |
| Common response normalization | Core service/adapter | Provides stable behavior across consumers where semantics genuinely match. |
| Provider-specific feature selection | Adapter plus explicit public configuration | Provider capabilities differ and should remain visible. |
| Business workflow and sequence of steps | Application/module | Depends on the project's learning objective and domain. |
| Prompt specific to one task/domain | Application/module, unless proven shared | Avoids turning core into a repository of unrelated prompts. |
| Embedding request/response contract | Core, when implemented | Shared capability useful to multiple consumers. |
| Vector-store selection and index policy | Application configuration plus adapter | Backend and domain requirements may vary. |
| Retrieval result representation | Core, where shared and feasible | Common document/source shape can reduce duplication. |
| What retrieved text is trusted and how it is cited | Application/workflow policy | Requires task-specific trust and user-facing decisions. |
| Memory storage/retrieval primitives | Core, when implemented | Reusable mechanism with explicit scope and contract. |
| Which memories to store or inject for a particular task | Application policy or an explicitly shared higher-level component | The policy is context-dependent. |
| Tool schema validation primitives | Core, if cross-project need is proven | Shared validation can be reused safely. |
| Tool authorization and approval | Application/security boundary | Must not be granted by model output alone. |
| Tool execution loop and stop policy | Application or intentionally designed orchestration component | Different workflows need different limits and semantics. |
| Provider secrets | Runtime configuration/secret source | Never hard-coded or committed. |
| UI, API routes, deployment, end-user messages | Application | Not part of a reusable GenAI core package. |

## 19. Implementation readiness checklist

Before calling the first LLM service milestone usable, verify:

- [x] The package can be imported from a consumer project using the documented workflow.
- [x] The supported public API and required configuration are documented.
- [x] Invalid requests and invalid configuration fail before unnecessary external calls.
- [x] Provider-specific request/response translation is isolated behind a defined integration boundary.
- [x] Successful results have a predictable normalized response shape.
- [x] Provider errors are translated into useful, safe error categories.
- [x] Unsupported settings/capabilities are reported rather than silently ignored.
- [x] Timeouts and retry behavior are bounded and documented.
- [x] Unit tests run without live credentials or network access.
- [x] Integration tests are separately controllable (gated on `INTEGRATION_TEST=1`).
- [x] Secrets and sensitive request content are not logged by default; automated secret scrubbing is active.
- [x] At least one consumer example uses public imports and does not depend on provider internals.
- [x] The core package does not import from its consuming modules.
- [x] Future/planned sections have not been mistaken for implemented features.

## 20. Rule for adding a new component

Before adding a component or promoting existing project code into `genai-core`, write down:

1. The concrete consumer/use case that requires it.
2. What responsibility it owns—and what it explicitly does not own.
3. Its public inputs, outputs, errors, configuration, and side effects.
4. The provider/storage/framework differences that must remain visible.
5. How it can be tested without relying on external services for every test.
6. Which existing projects will use it and how their current code will migrate.
7. The maintenance and dependency cost introduced by adding it.

Implement the smallest contract that addresses the use case. Keep the component local if its reusable boundary is not yet clear. Update this document when the behavior changes so that it describes the actual supported contract, not only the original intention.

## 21. Relationship to `GOAL.md`

`GOAL.md` answers why `genai-core` exists, what outcomes matter, what its boundaries are, and how it should evolve.

This document converts that intent into behavioral expectations: who owns each responsibility, how requests and failures should behave, how components compose, and which future capabilities are not yet commitments. If implementation reveals a conflict between the documents, resolve and document the design decision rather than allowing the two specifications to drift.
