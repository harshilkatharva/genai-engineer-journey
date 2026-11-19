# Module 20: Parallel Research Team — Execution & Data Flow

This document details how data and control flow through the implemented components of Module 20. It reflects the concrete implementation as built.

---

## 1. Domain Model Hierarchy & Relationships

```mermaid
classDiagram
    class ResearchDecomposition {
        +str topic
        +EffortLevel effort_level
        +str rationale
        +int worker_count
        +list[WorkerBrief] worker_briefs
    }

    class WorkerBrief {
        +str brief_id
        +str title
        +str topic
        +str objective
        +BriefType brief_type
        +str worker_type
        +list[str] depends_on
        +list[str] task_boundaries
        +OutputFormatSpec expected_output_format
        +list[str] suggested_tools
        +int max_steps
        +float timeout_seconds
    }

    class OutputFormatSpec {
        +str schema_name
        +list[str] required_fields
        +str guidelines
    }

    class WorkerExecutionResult {
        +str brief_id
        +WorkerStatus status
        +WorkerResearchOutput output
        +int tokens_used
        +float cost_usd
        +float duration_ms
        +str error_message
    }

    class WorkerResearchOutput {
        +str brief_id
        +str findings_summary
        +list[dict] claims
        +list[dict] sources
        +list[str] search_queries_used
        +list[str] limitations
    }

    class CheckpointState {
        +str run_id
        +str user_id
        +str topic
        +RunStatus status
        +ResearchDecomposition decomposition
        +dict[str, WorkerExecutionResult] worker_results
        +dict[str, dict] authoritative_sources
        +int used_tokens
        +float used_cost_usd
        +float elapsed_seconds
        +dict final_report
    }

    ResearchDecomposition "1" *-- "1..*" WorkerBrief
    WorkerBrief "1" *-- "1" OutputFormatSpec
    WorkerExecutionResult "1" *-- "0..1" WorkerResearchOutput
    CheckpointState "1" o-- "0..1" ResearchDecomposition
    CheckpointState "1" *-- "0..*" WorkerExecutionResult
```

---

## 2. Shared Budget Concurrency Flow

The `SharedRunBudget` coordinates concurrent worker tasks and prevents race conditions using an `asyncio.Lock`:

```mermaid
sequenceDiagram
    autonumber
    actor Worker1 as Parallel Worker 1
    actor Worker2 as Parallel Worker 2
    participant Budget as SharedRunBudget
    participant Lock as asyncio.Lock

    Worker1->>Budget: can_proceed()
    activate Budget
    Budget->>Lock: acquire()
    Lock-->>Budget: lock acquired
    Budget->>Budget: Check deadline, tokens, cost
    Budget->>Lock: release()
    Budget-->>Worker1: True
    deactivate Budget

    Worker2->>Budget: can_proceed()
    activate Budget
    Budget->>Lock: acquire()
    Lock-->>Budget: lock acquired
    Budget->>Budget: Check deadline, tokens, cost
    Budget->>Lock: release()
    Budget-->>Worker2: True
    deactivate Budget

    par Concurrent Execution
        Worker1->>Worker1: Execute LLM completion (tokens=350, cost=$0.001)
    and
        Worker2->>Worker2: Execute LLM completion (tokens=420, cost=$0.0012)
    end

    Worker1->>Budget: record_usage(tokens=350, cost=$0.001)
    activate Budget
    Budget->>Lock: acquire()
    Budget->>Budget: current_tokens += 350, current_cost_usd += $0.001
    Budget->>Lock: release()
    deactivate Budget

    Worker2->>Budget: record_usage(tokens=420, cost=$0.0012)
    activate Budget
    Budget->>Lock: acquire()
    Budget->>Budget: current_tokens += 420, current_cost_usd += $0.0012
    Budget->>Lock: release()
    deactivate Budget
```

---

## 3. Atomic Checkpoint Persistence Flow

`FileCheckpointStore` guarantees crash consistency by writing snapshots to a temporary file before atomically replacing the destination:

```mermaid
sequenceDiagram
    autonumber
    participant Caller as Orchestrator / Scheduler
    participant Store as FileCheckpointStore
    participant FS as Local Filesystem

    Caller->>Store: save_checkpoint(state)
    activate Store
    Store->>Store: Acquire async lock
    Store->>FS: Create NamedTemporaryFile (tmp_cp_<run_id>_*.json)
    Store->>FS: Write state.model_dump_json(indent=2)
    Store->>FS: flush() and os.fsync() to disk
    Store->>FS: os.replace(temp_path, checkpoint_<run_id>.json)
    Store->>Store: Release async lock
    Store-->>Caller: None (Success)
    deactivate Store
```

---

## 4. Lead Agent Decomposition Flow

The `DecompositionEngine` orchestrates memory recall, prompt rendering, structured LLM generation, validation, and goal pinning:

```mermaid
sequenceDiagram
    autonumber
    participant Client as Orchestrator / Caller
    participant Decomp as DecompositionEngine
    participant LTM as LongTermMemory
    participant Prompts as PromptTemplateManager
    participant LLM as LLMService
    participant Validator as DecompositionValidator
    participant STM as ShortTermMemory

    Client->>Decomp: decompose(topic, user_id, session_id, effort_override)
    activate Decomp

    opt Memory Integration Enabled
        Decomp->>LTM: search(user_id=user_id, query=topic, top_k=3)
        LTM-->>Decomp: Recalled preferences / constraints
    end

    Decomp->>Prompts: render_decomposition_prompts(topic, max_workers, default_effort, recalled_context)
    Prompts-->>Decomp: (system_prompt, user_prompt)

    Decomp->>LLM: complete_structure(messages, response_schema=ResearchDecomposition)
    LLM-->>Decomp: Structured ResearchDecomposition plan

    Decomp->>Validator: validate(plan, max_workers, effort_override)
    Validator-->>Decomp: Validation passed

    opt Short-Term Memory Enabled
        Decomp->>STM: pin_goal(user_id, session_id, goal_text, metadata)
        STM-->>Decomp: Goal pinned
    end

    Decomp-->>Client: Validated ResearchDecomposition
    deactivate Decomp
```

---

## 5. Bounded Research Worker Execution & Evidence Collection Flow

`DefaultResearchWorker` wraps `ToolLoopRunner`, collects evidence deterministically, deducts shared budget atomically, and isolates all failures:

```mermaid
sequenceDiagram
    autonumber
    participant Scheduler as Scheduler / Orchestrator
    participant Worker as DefaultResearchWorker
    participant Budget as SharedRunBudget
    participant Collector as EvidenceCollector
    participant Runner as ToolLoopRunner
    participant Tool as search_web / search_kb
    participant Extractor as EvidenceExtractor

    Scheduler->>Worker: execute(brief, shared_budget)
    activate Worker

    Worker->>Budget: can_proceed()
    Budget-->>Worker: True

    Worker->>Collector: Create fresh EvidenceCollector()
    Worker->>Runner: run(messages, budget=ToolBudget(max_steps, timeout))
    activate Runner

    loop Tool Iterations (max_steps <= 5)
        Runner->>Tool: execute(query)
        activate Tool
        Tool->>Collector: add_source(source_id, chunk_id, content, ...)
        Collector-->>Tool: SourceReference recorded
        Tool-->>Runner: Formatted text with Source ID and Chunk ID
        deactivate Tool
    end

    Runner-->>Worker: ToolLoopResult(status, final_text, messages)
    deactivate Runner

    Worker->>Budget: record_usage(tokens=total_tokens, cost_usd=cost)
    Budget-->>Worker: None (Atomic update)

    Worker->>Extractor: extract(brief, final_text, collector)
    Extractor-->>Worker: WorkerResearchOutput(sources, claims, findings)

    Worker-->>Scheduler: WorkerExecutionResult(status=COMPLETED/PARTIAL, output=...)
    deactivate Worker
```

---

## 6. Parallel Scheduling, Concurrency Control & Real-Time Checkpointing Flow

`ParallelResearchScheduler` orchestrates concurrent worker execution through `asyncio.Semaphore`, isolates sibling failures with `asyncio.gather(..., return_exceptions=True)`, validates deliverables, and persists state incrementally:

```mermaid
sequenceDiagram
    autonumber
    participant Orchestrator as Orchestrator / Lead Agent
    participant Scheduler as ParallelResearchScheduler
    participant Semaphore as asyncio.Semaphore(max_concurrency)
    participant Worker as ResearchWorker
    participant Validator as WorkerOutputValidator
    participant Store as CheckpointStore

    Orchestrator->>Scheduler: schedule_and_execute(briefs, budget, checkpoint)
    activate Scheduler

    Scheduler->>Scheduler: Check dependency cycles

    par Concurrent Execution (throttled by Semaphore)
        Scheduler->>Semaphore: acquire()
        Semaphore-->>Scheduler: slot granted
        Scheduler->>Worker: execute(brief_1, budget)
        activate Worker
        Worker-->>Scheduler: WorkerExecutionResult (COMPLETED)
        deactivate Worker
        Scheduler->>Semaphore: release()
        Scheduler->>Validator: validate(result_1, brief_1)
        Validator-->>Scheduler: Validated result
        Scheduler->>Store: save_checkpoint(checkpoint with result_1 & sources)
    and
        Scheduler->>Semaphore: acquire()
        Semaphore-->>Scheduler: slot granted
        Scheduler->>Worker: execute(brief_2, budget)
        activate Worker
        Worker-->>Scheduler: WorkerExecutionResult (FAILED / TIMEOUT)
        deactivate Worker
        Scheduler->>Semaphore: release()
        Scheduler->>Validator: validate(result_2, brief_2)
        Validator-->>Scheduler: Validated result (partial sources salvaged)
        Scheduler->>Store: save_checkpoint(checkpoint with result_2 & salvaged sources)
    end

    Scheduler-->>Orchestrator: dict[brief_id, WorkerExecutionResult]
    deactivate Scheduler
```
