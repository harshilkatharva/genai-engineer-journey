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
