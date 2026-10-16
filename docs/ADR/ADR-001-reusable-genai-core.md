# ADR-001: Establish `genai-core` as a Reusable GenAI Foundation

- **Status:** Accepted
- **Decision Date:** 2026-10-16
- **Decision Scope:** Repository architecture, shared infrastructure, and future GenAI projects
- **Initial Consumers:** Modules 17–20
- **Future Consumers:** Other learning modules and independent projects

---

## 1. Context

The `genai-engineer-journey` repository contains a sequence of learning modules and mini-projects covering Python, software engineering, LLMs, prompt engineering, APIs, embeddings, vector databases, RAG, LangChain, LangGraph, LlamaIndex, MCP, tool calling, function calling, memory, and AI agents.

Modules 0–16 have already been developed. Across these modules, useful implementations have been built for different GenAI capabilities.

However, individual mini-projects may contain functionality that later projects also need. Reusing that functionality can require copying code, adjusting imports, resolving dependencies, adapting interfaces, and understanding implementation details from another project.

For example:

- An LLM provider implemented in an earlier project may be needed by a later agent.
- An embedding implementation may be needed by multiple retrieval-based applications.
- A retriever may be reused by different RAG and agent applications.
- A memory implementation may be needed by multiple conversational systems.
- Evaluation utilities may be needed across different retrieval, generation, and agent workflows.

Without a shared foundation, each new mini-project risks becoming responsible for rebuilding or maintaining functionality that already exists elsewhere.

This problem becomes increasingly significant as projects grow more complex and begin combining multiple capabilities.

Before starting Module 17, we therefore decided to establish a shared package that can provide reusable GenAI infrastructure for the upcoming modules and future projects.

## 2. Problem Statement

How can we reuse existing implementations and build future GenAI capabilities without repeatedly copying code between mini-projects, creating inconsistent interfaces, or introducing unnecessary dependencies between projects?

The solution should provide a shared foundation while preserving the independent nature of individual learning modules.

It must also balance two competing concerns:

1. **Reusability:** Shared functionality should be implemented once and made available to multiple consumers.
2. **Simplicity:** The shared package must not become an unnecessarily complex framework that slows down development or forces unrelated projects to use the same architecture.

The objective is not to move every existing project into a common codebase. It is to identify genuinely reusable capabilities and provide a maintainable place for them.

## 3. Decision

We will create a shared Python package:

`packages/genai-core/`

This package will serve as the reusable foundation for common GenAI functionality used by Modules 17–20 and other projects in the future.

**The package will be established before starting Module 17.**

Rather than implementing every possible component upfront, development will proceed incrementally.

The initial work starts with integrating the LLM-related functionality already implemented in `LLMService`. After establishing that foundation, we will progressively introduce additional reusable capabilities such as embeddings, retrieval, memory, and evaluation.

Each capability will be added when its requirements are sufficiently understood and its reusable value is justified.

The core principle is:

> Build reusable infrastructure from existing, understood implementations; introduce capabilities incrementally; and keep module-specific application logic outside the shared package.

## 4. Why Are We Creating `genai-core`?

### 4.1 Eliminate repeated code copying

Future mini-projects should be able to import reusable functionality from `genai-core` instead of copying implementations from earlier modules.

When a shared component needs a fix or improvement, that change can be made in the shared package and tested in one place.

Consumers can then use the updated component through its supported interface rather than maintaining separate copies.

### 4.2 Solve import and integration problems

Existing mini-projects may use different folder structures, imports, configuration conventions, and dependency arrangements.

Code that works inside one project may not be directly reusable in another.

The shared package will provide a proper package boundary, consistent imports, controlled dependencies, and documented interfaces.

This makes reuse a deliberate architectural capability rather than an exercise in copying files until the imports work.

### 4.3 Establish consistent interfaces

Similar capabilities should expose consistent interfaces where doing so is useful.

For example, consuming applications should not need to know every provider-specific implementation detail to perform a supported LLM operation.

Likewise, future embedding, retrieval, and memory components should expose clear contracts so they can be combined without depending on each other's internal implementation.

Consistency does not mean forcing every component to behave identically when underlying technologies have different capabilities.

### 4.4 Make components independently testable

Shared infrastructure should have tests that verify its contracts, expected behavior, error handling, and integration boundaries.

This reduces the risk that a change to one shared component unexpectedly breaks multiple consuming projects.

### 4.5 Prepare for increasingly connected modules

Modules 17–20 are the next stage of the learning journey. Their projects may require multiple capabilities to work together.

Instead of implementing separate versions of LLM access, embeddings, retrieval, memory, and evaluation wherever they are needed, the modules can build on a common foundation.

The exact components each module consumes will be determined by its actual requirements.

### 4.6 Support reuse beyond the learning repository

`genai-core` should not be designed solely around one mini-project or one module.

Its interfaces and dependency structure should make reuse in other projects possible when those projects need the same capabilities.

Independent reuse does not mean adding every possible feature now. It means avoiding unnecessary coupling to an individual module.

## 5. Why Must It Be Built Before Module 17?

Building the shared package before starting Module 17 establishes a reusable foundation at the beginning of the next development stage.

Without this preparation, Module 17 might implement its own versions of components that Module 18, 19, or 20 later need. Subsequent modules would then require copying, reconciling, or replacing those implementations.

By establishing `genai-core` first:

1. Existing reusable functionality can be extracted and tested before new modules depend on it.
2. Module 17 can begin with a consistent foundation.
3. Later modules can reuse capabilities instead of repeatedly rebuilding infrastructure.
4. New shared functionality can be improved centrally as new requirements emerge.
5. The learning modules can focus on their primary concepts rather than repeatedly solving packaging and integration problems.

The package may initially contain only a limited set of capabilities. That is intentional. Establishing a clean boundary is more important than immediately implementing a large number of components.

## 6. Existing Code and Extraction Strategy

The initial extraction will use selected implementations from:

- `module-09-rag`
- `module-15-function_calling`

Module 09 provides existing LLM-related and structured-output functionality, while Module 15 provides existing LLM-related and tool/function-calling functionality.

These implementations will serve as source material for the first shared LLM capability.

The extraction process must inspect the actual code, dependencies, interfaces, behavior, and tests before deciding what to reuse directly and what to adapt.

The objective is to create a consistent implementation inside `genai-core`, not merely copy files into a new directory without resolving their existing coupling.

### Preservation decision

Modules 0–16 will remain unchanged.

- Existing code will be extracted or adapted into `packages/genai-core/`.
- Changes needed for reusability will be made inside the new package.
- Existing implementations will remain available as original learning projects and references.
- Existing modules will not be migrated to the shared package as part of this decision.

The package must be capable of operating independently of the original modules' internal import paths.

As additional capabilities are introduced, code from other existing modules may be considered as a source when explicitly needed. Such extraction must follow the same preservation principle.

## 7. Incremental Development Roadmap

`genai-core` will grow through deliberate implementation stages rather than a single large extraction.

<escape>
</escape>

### Stage 1 — Foundation and LLM Service

**Purpose:** Establish the basic package structure and a reusable interface for LLM operations.

Initial responsibilities include:

- Shared LLM models and interfaces.
- `LLMService`.
- Provider adapters.
- Structured output.
- Tool calling.
- Asynchronous streaming where supported.
- Normalized response handling.
- Dependency injection.
- Package-level tests and installation configuration.

The existing implementations from Module 09 and Module 15 will be inspected and adapted for this stage.

This stage establishes the first shared capability; it does not define the entire scope of `genai-core`.

### Stage 2 — Embedding

**Purpose:** Provide reusable functionality for converting supported text inputs into vector representations.

Potential responsibilities include:

- Embedding interfaces.
- Provider adapters where appropriate.
- Consistent input and output representations.
- Batch processing where supported.
- Configuration and error handling.
- Tests for dimensions, output validation, and provider behavior.

Embedding functionality may later be consumed by semantic search, retrieval, RAG, and memory implementations.

### Stage 3 — Retrieval

**Purpose:** Provide reusable retrieval interfaces and implementations for finding relevant information.

Potential responsibilities include:

- Retriever interfaces.
- Query and retrieval result models.
- Integration with supported vector stores or retrieval backends.
- Metadata and filtering contracts.
- Ranking or reranking integration when justified.
- Tests for retrieval behavior and result consistency.

The package should distinguish generic retrieval contracts from backend-specific implementations.

Retrieval should not automatically absorb all RAG application logic. Individual applications should retain control over how retrieved context is assembled and used.

### Stage 4 — Memory

**Purpose:** Make reusable memory capabilities available to future conversational and agentic systems.

Potential responsibilities include:

- Memory interfaces.
- Memory record models.
- Supported storage and retrieval integrations.
- Relevant context retrieval.
- Persistence and lifecycle behavior.
- Tests for storing, retrieving, and updating memory.

Memory should build on appropriate shared capabilities where useful, while maintaining a clear distinction between general storage, semantic retrieval, and application-specific memory policies.

The existing Memory module may serve as a source for future extraction when required.

### Stage 5 — Evaluation

**Purpose:** Support repeatable evaluation of reusable GenAI components and complete application pipelines.

Potential responsibilities include:

- Shared evaluation models.
- Evaluation interfaces.
- Metrics and evaluator abstractions.
- Dataset and result representations.
- Evaluation of retrieval and generation.
- Traceable evaluation outputs.
- Tests for metric computation and evaluation behavior.

Evaluation capabilities may later help assess whether changes to retrieval, generation, memory, or agent workflows improve or degrade system quality.

The initial implementation should focus on useful, testable evaluation contracts rather than introducing a large evaluation framework prematurely.

### Roadmap principle

These stages describe the intended direction, not a commitment to implement everything immediately or in exactly this order.

Each stage will be refined as actual project requirements emerge. New capabilities will be added only when their responsibilities, dependencies, interfaces, and expected consumers are understood.

## 8. How `genai-core` Will Be Used by Future Modules

The intended relationship is:

```text
Existing implementations
(module-09, module-15, and others as needed)
                    |
                    v
       Extract and adapt reusable code
                    |
                    v
             packages/
              genai-core
                    |
        +-----------+-----------+
        |           |           |
        v           v           v
     Module 17   Module 18   Module 19
        |           |           |
        +-----------+-----------+
                    |
                    v
                Module 20
                    |
                    v
       Other projects in the future
```

This illustrates the intended reuse relationship; it is not a requirement that every module depend on every component or that development follow a strict linear sequence.

Examples of future reuse include:

| Future capability | Potential use in later projects |
|---|---|
| LLM service | Consistent model access for applications and agents |
| Embeddings | Semantic search, vector storage, RAG, and memory |
| Retrieval | Finding relevant documents or contextual information |
| Memory | Reusing context across conversational interactions |
| Evaluation | Comparing retrieval and generation quality and validating system changes |

When a new module needs one of these capabilities, it should use the supported shared interface when appropriate.

The module should still own its application-specific logic, learning objectives, configuration, and orchestration.

A module-specific requirement should not automatically become a responsibility of `genai-core`. It should be promoted into the shared package only when a reusable abstraction offers a clear benefit.

## 9. Architectural Boundaries

To remain useful as the package grows, `genai-core` must have clear boundaries.

### 9.1 Shared infrastructure versus application logic

`genai-core` owns reusable capabilities, contracts, and integrations.

Individual modules own their specific workflows, business rules, application behavior, and learning exercises.

For example, the package may provide an LLM service, but an AI-agent project determines when and why that service is invoked.

The package may provide a retriever, but a RAG application determines how retrieved information is incorporated into a particular response.

### 9.2 Shared interfaces versus implementation details

Consumers should rely on supported interfaces instead of importing arbitrary internal files or depending on provider SDK details.

Internal implementations may change when necessary, but intentional interface changes must be assessed for their effect on consumers.

### 9.3 Optional capabilities versus mandatory dependencies

Not every project will need every component.

The package structure and dependency design should avoid forcing a simple LLM-only project to install or configure unrelated memory, vector database, or evaluation dependencies when they can reasonably be optional.

Dependencies should be introduced when required by a capability, not merely because that capability might be implemented in the future.

### 9.4 Incremental abstraction versus premature generalization

A component should be generalized when actual implementations or consumers demonstrate a useful common contract.

Do not create complicated abstractions solely to anticipate hypothetical providers, storage systems, agents, or workflows.

### 9.5 Clear failure and capability behavior

Shared interfaces must not pretend that every provider or backend supports identical operations.

Unsupported capabilities should be explicit, and errors should be understandable to consumers.

Normalization must preserve meaningful information rather than silently discarding provider-specific details needed by applications.

## 10. Complexity Introduced by `genai-core`

Creating a shared package is not free. It moves some complexity away from individual mini-projects and into a central component that must be designed and maintained carefully.

The following costs are accepted because they may reduce duplication and integration overhead as the repository grows.

### 10.1 Initial extraction cost

Existing code must be inspected, dependencies identified, interfaces reconciled, and tests created.

This takes time that could otherwise be spent implementing the next mini-project.

**Mitigation:** Start with a focused LLM integration and expand only when a genuine need appears.

### 10.2 Additional abstraction layers

A consumer may call a shared service, which delegates through an interface to a provider or backend adapter.

This improves consistency but introduces additional files, classes, and navigation when compared with directly calling an SDK.

**Mitigation:** Use abstractions only where they meaningfully improve reuse, testability, or replaceability.

### 10.3 Shared contract maintenance

A change to a public interface may affect several modules or external projects.

The shared package therefore requires stronger attention to compatibility, testing, and documentation than an isolated mini-project.

**Mitigation:** Define clear contracts, test their expected behavior, and review the impact of breaking changes.

### 10.4 Dependency management

Multiple capabilities may need different third-party libraries. A shared package can become difficult to install if every integration is mandatory.

**Mitigation:** Keep dependencies explicit and use optional dependency groups where appropriate.

### 10.5 Testing burden

Shared components must be validated independently and, where appropriate, against their integration points.

A successful unit test alone may not prove that a complete future application works correctly.

**Mitigation:** Test shared contracts and implementations independently, then add focused integration tests in consuming projects.

### 10.6 Risk of becoming a large, tightly coupled package

If every feature from every module is added automatically, `genai-core` could become a difficult-to-maintain collection of unrelated functionality.

**Mitigation:** Keep component boundaries clear, avoid unnecessary cross-dependencies, and require a concrete justification for additions.

### 10.7 Centralized change impact

A bug in a heavily reused component could affect several projects simultaneously.

**Mitigation:** Add appropriate automated tests, preserve compatibility where practical, communicate important changes, and run relevant tests for affected consumers.

### Overall complexity assessment

The package deliberately adds some upfront architecture and maintenance responsibilities in exchange for more consistent reuse.

The decision is beneficial only if the package stays smaller and clearer than the cumulative duplication and integration work it replaces.

Reusability is the objective; abstraction for its own sake is not.

## 11. Expected Future Impact

### Positive impacts

- **Less duplicated code:** Common implementations can be maintained centrally.
- **Faster module development:** New projects can start with established capabilities.
- **More consistent behavior:** Shared contracts reduce avoidable differences between implementations.
- **Easier testing:** Common functionality can be validated independently.
- **Better integration:** Clearly defined interfaces make combinations of components easier to reason about.
- **Controlled evolution:** New capabilities can be introduced incrementally.
- **Improved learning quality:** Future modules can focus more on their primary topics instead of repeatedly rebuilding infrastructure.
- **Potential external reuse:** Other projects can adopt the package when its interfaces and dependencies meet their needs.

### Risks and limitations

- Not every existing implementation will be reusable without adaptation.
- The shared package can introduce unnecessary complexity if its scope is not controlled.
- Provider and backend differences may prevent complete uniformity.
- Changes to shared interfaces may require updates in multiple consumers.
- A reusable component still needs application-level integration, configuration, and testing.
- Introducing a shared package does not automatically improve the quality of every system built on it.

These risks will be managed through incremental development, explicit boundaries, testing, and careful decisions about what belongs in the shared foundation.

## 12. Alternatives Considered

### Alternative A — Continue implementing each mini-project independently

**Rejected.**

This keeps projects isolated but repeats work, increases integration effort, and creates multiple versions of similar functionality.

### Alternative B — Copy useful code from previous modules whenever needed

**Rejected.**

This may solve an immediate problem but preserves the difficulties of imports, dependencies, repeated fixes, and inconsistent interfaces.

### Alternative C — Move all existing module implementations into one shared framework immediately

**Rejected.**

This would increase the scope and risk of the extraction, create unnecessary abstractions, and make it harder to validate the resulting package.

### Alternative D — Build a focused shared package before Module 17 and expand it incrementally

**Accepted.**

This establishes reusable infrastructure early while keeping the initial scope manageable. It allows the package to grow based on real needs from later modules instead of hypothetical future requirements.

## 13. Implementation and Verification Principles

The implementation should follow these principles:

1. Inspect source code and dependencies before extracting a component.
2. Preserve the original implementations and their existing behavior.
3. Keep new reusable code within `packages/genai-core/` and its package-specific configuration and tests.
4. Define clear interfaces and keep provider- or backend-specific details isolated where appropriate.
5. Test components independently of their original modules.
6. Document architecture, execution flow, confirmed business rules, and significant design decisions.
7. Verify package installation and imports in an appropriate isolated environment.
8. Add new capabilities incrementally and verify their actual requirements.
9. Avoid implementing speculative features or overgeneralizing existing code.
10. Check test results and known limitations before declaring a capability ready for reuse.

No changes to Modules 0–16 are part of this architectural decision.

The original implementations remain the reference for their existing behavior, and the shared package will be developed independently.

## 14. Success Criteria

The architectural decision is successful when:

- [ ] `packages/genai-core/` exists as an independently installable Python package.
- [ ] The initial LLM functionality has been extracted and adapted from Module 09 and Module 15.
- [ ] The package exposes documented and testable interfaces.
- [ ] Shared components can be imported without depending on the original modules' internal import paths.
- [ ] Package tests validate the behavior of implemented capabilities.
- [ ] Modules 17–20 can reuse applicable shared capabilities without copying their implementations.
- [ ] Additional capabilities can be introduced without making unrelated components mandatory dependencies.
- [ ] The package's responsibilities and module-specific responsibilities remain distinct.
- [ ] The complexity of the shared package remains justified by actual reuse.
- [ ] No files in Modules 0–16 have been modified as part of the extraction.

These criteria should be marked complete only after the corresponding implementation and verification have actually occurred.

## 15. Reassessment Conditions

This decision should be reassessed if:

- The shared package becomes a source of excessive coupling.
- Its abstractions repeatedly make simple use cases harder.
- Dependencies become difficult to manage.
- Multiple consumers require incompatible interfaces.
- Shared changes cause frequent regressions.
- A capability proves to be genuinely application-specific rather than reusable.
- A clearer package boundary is needed to support independent release or maintenance.

Reassessment does not automatically mean abandoning the shared package. It means evaluating whether its boundaries, dependency structure, or public interfaces need refinement.

## 16. Final Decision Summary

Before starting Module 17, we will establish `packages/genai-core/` as the reusable foundation for common GenAI infrastructure.

The first implementation step is to integrate and adapt the existing LLM service functionality using selected code from Modules 09 and 15. Subsequent capabilities, including embeddings, retrieval, memory, and evaluation, will be added incrementally as their requirements become clear.

The purpose of the package is to reduce code duplication, avoid repeated integration problems, establish consistent interfaces, improve independent testing, and enable future modules and projects to build on existing work.

This architecture deliberately introduces some shared abstraction and maintenance complexity. That cost is accepted only where the benefits of reuse, consistency, and maintainability justify it.

The original modules remain unchanged, application-specific logic stays within its consuming projects, and the shared package grows only when a concrete reusable need warrants it.

**Final architectural principle: Establish the shared foundation before Module 17, reuse existing work, add capabilities step by step, and keep the complexity of `genai-core` proportional to the value it provides.**