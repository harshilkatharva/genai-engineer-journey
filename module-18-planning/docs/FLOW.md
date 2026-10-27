# Module 18: Plan-and-Execute Research Assistant — Execution & Data Flow

## 1. End-to-End System Execution Flow

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Operator
    participant Assistant as PlanAndExecuteAssistant
    participant Mem as PlanningMemoryManager
    participant Planner as PlanGenerator
    participant Validator as PlanValidator
    participant Reviewer as HumanPlanReviewGateway
    participant Scheduler as DAGScheduler
    participant Executor as Module17StepExecutor
    participant Verifier as StepVerifier
    participant Replanner as ReplanningEngine
    participant Store as RunRecordStore

    User->>Assistant: Submit Goal (Investigate customer C-101 and refund ticket T-101)
    
    rect rgb(240, 245, 255)
    Note over Assistant,Mem: Phase 1: Context and Memory Retrieval
    Assistant->>Mem: get_user_preferences(user_id, query=goal)
    alt enable_episodic_similarity_search is True
        Assistant->>Mem: get_similar_past_plans(user_id, goal)
        Mem-->>Assistant: Recalled past plan episodes
    else enable_episodic_similarity_search is False
        Assistant->>Mem: Bypasses episodic retrieval (Zero-Shot)
    end
    Mem-->>Assistant: Long-term preferences and past plan blueprints
    end

    rect rgb(245, 240, 255)
    Note over Assistant,Validator: Phase 2: Plan Generation and Automated Repair Loop
    Assistant->>Executor: get_available_capabilities()
    Executor-->>Assistant: Capability map (names and descriptions)
    Assistant->>Planner: generate_plan(goal, context, preferences, past_plans, capabilities)
    Planner->>Planner: complete_structure(Plan) via LLMService
    Planner->>Validator: validate_plan(plan)
    alt Validation Fails
        Validator-->>Planner: PlanValidationError (Cycle trace or missing ID)
        Planner->>Planner: PlanRepairer.repair_plan(invalid_plan, error, retries)
        Planner->>Validator: Re-validate repaired plan
    end
    Validator-->>Assistant: Validated Plan v1 (DAG confirmed)
    Assistant->>Store: save_plan_version(run_id, v1, author=planner_llm)
    end

    rect rgb(255, 245, 235)
    Note over Assistant,Reviewer: Phase 3: Human Plan Review (Gating)
    Assistant->>Reviewer: request_review(Plan v1)
    Reviewer->>User: Present Plan v1 (steps, dependencies, criteria)
    alt Operator Modifies Plan
        User-->>Reviewer: submit_decision(MODIFY, modified_plan)
        Reviewer->>Validator: validate_plan(modified_plan)
        Reviewer->>Store: save_plan_version(run_id, v2, author=human_reviewer)
    else Operator Rejects Plan
        User-->>Reviewer: submit_decision(REJECT, reason=Unsafe strategy)
        Reviewer-->>Assistant: Abort run (RunStatus.CANCELLED)
    else Operator Approves Plan
        User-->>Reviewer: submit_decision(APPROVE)
    end
    end

    rect rgb(240, 255, 245)
    Note over Assistant,Scheduler: Phase 4: Concurrent DAG Scheduling and Step Execution
    Assistant->>Scheduler: execute_plan(approved_plan, budget=RunBudget)
    loop While Unfinished Steps Exist
        Scheduler->>Scheduler: resolve_ready_steps()
        critical Step Execution Bounded by Semaphore
            Scheduler->>Executor: execute_step(step, predecessor_context)
            Executor->>Executor: BoundedAgent.run(step)
            Executor-->>Scheduler: StepExecutionResult(status, text, tokens, cost)
            
            Scheduler->>Verifier: verify_step(step, result)
            Verifier->>Verifier: Gate 1 Programmatic and Gate 2 Model Judge
            Verifier-->>Scheduler: StepVerificationResult(passed, reasoning)
        end

        alt Step Verification Passes
            Scheduler->>Scheduler: Mark step COMPLETED and cache deliverable
        else Verification Fails and Attempts under max_step_retries
            Scheduler->>Scheduler: Retry step with verifier feedback
        else Verification Fails and Attempts Exhausted
            Scheduler->>Replanner: trigger_replan(plan, completed_steps, failed_step)
            Note over Replanner: Settle in-flight siblings and preserve completed steps
            Replanner->>Validator: validate_plan(revised_plan)
            Replanner->>Store: save_plan_version(run_id, v_next, author=replanner)
            Replanner-->>Scheduler: Updated Plan resuming unfulfilled steps
        end
    end
    Scheduler-->>Assistant: All Steps Completed
    end

    rect rgb(245, 245, 255)
    Note over Assistant,Store: Phase 5: Synthesis and Persistence
    Assistant->>Assistant: Synthesize final verified deliverable
    Assistant->>Mem: record_episode(action_type=plan_execution, goal, plan_summary)
    Assistant->>Store: finalize_run_record(run_id, deliverable, total_tokens, total_cost)
    Assistant-->>User: Final Verified Deliverable + Execution Summary
    end
```

---

## 2. Structural DAG Validation Flow (Milestone 1 Implementation)

```mermaid
flowchart TD
    A["Raw Plan Input"] --> B{"plan.steps empty?"}
    B -- "Yes" --> B1["Raise EmptyPlanError"]
    B -- "No" --> C{"Duplicate or Empty IDs?"}
    C -- "Yes" --> C1["Raise DuplicateStepIdError / PlanValidationError"]
    C -- "No" --> D{"Self Dependencies?"}
    D -- "Yes" --> D1["Raise SelfDependencyError"]
    D -- "No" --> E{"Missing Dependencies in plan?"}
    E -- "Yes" --> E1["Raise MissingDependencyError"]
    E -- "No" --> F["3-State Depth-First Search Cycle Detection"]
    F --> G{"Directed Cycle Encountered?"}
    G -- "Yes" --> G1["Raise CircularDependencyError with Cycle Path"]
    G -- "No" --> H{"Root Step Exists with No Dependencies?"}
    H -- "No" --> H1["Raise UnreachablePlanError"]
    H -- "Yes" --> I["Validation Succeeded: Valid DAG"]
```

---

## 3. Plan Generation, Automated Repair & Memory Integration Flow (Milestone 2 Implementation)

```mermaid
flowchart TD
    Start["User Submits Goal"] --> MemLTM["Query Long-Term Preferences via search_longterm"]
    Start --> MemCheck{"enable_episodic_similarity_search?"}
    MemCheck -- "True" --> MemEpisodic["Recall Past Plan Episodes"]
    MemCheck -- "False" --> ZeroShot["Skip Past Plan Retrieval / Zero-Shot"]
    
    MemLTM --> CapQuery["Query StepExecutor.get_available_capabilities"]
    MemEpisodic --> CapQuery
    ZeroShot --> CapQuery

    CapQuery --> PromptGen["Build Planning Prompt with Goal, Capabilities, Preferences and Few-Shot Blueprints"]
    PromptGen --> LLMGen["LLMService.complete_structure(Plan)"]
    
    LLMGen --> Validate{"PlanValidator.validate_plan"}
    Validate -- "Valid DAG" --> ReturnPlan["Return Validated Plan v1"]
    
    Validate -- "PlanValidationError" --> RepairCheck{"Attempt <= max_planning_retries?"}
    RepairCheck -- "Yes" --> BuildRepairPrompt["Build Repair Prompt with Diagnostic Stack Trace"]
    BuildRepairPrompt --> LLMRepair["LLMService.complete_structure(Plan)"]
    LLMRepair --> Validate
    
    RepairCheck -- "No" --> FailPlan["Raise PlanGenerationError (retries exhausted)"]
```

---

## 4. Async DAG Execution & Dual Verification Flow (Milestone 4 Implementation)

```mermaid
flowchart TD
    Start["DAGScheduler.execute_plan"] --> TopoCheck["Identify READY Steps: All depends_on in step_results"]
    TopoCheck --> ActiveCheck{"All Steps COMPLETED or FAILED?"}
    ActiveCheck -- "Yes" --> FinalResult["Return SchedulerExecutionResult"]
    ActiveCheck -- "No" --> SiblingCheck{"Failed Step Detected?"}
    
    SiblingCheck -- "Yes" --> SettleWait["Await Active Sibling Tasks via asyncio.gather"]
    SettleWait --> RecordSiblings["Cache Successful Sibling Outputs"]
    RecordSiblings --> ReturnFailed["Return Status FAILED with failed_step_id"]

    SiblingCheck -- "No" --> ConcurrencyGate["Acquire asyncio.Semaphore(max_concurrency)"]
    ConcurrencyGate --> RunStep["Module17StepExecutor.execute_step with predecessor_context"]
    
    RunStep --> Gate1{"Gate 1: Programmatic Verifier"}
    Gate1 -- "Status!=COMPLETED or Text < 20 chars" --> Gate1Fail["Programmatic Check Failed"]
    Gate1 -- "Passed" --> Gate2{"Gate 2: LLM Judge Verifier"}
    
    Gate2 --> ScoreEval{"is_satisfied and score >= 0.85?"}
    ScoreEval -- "Passed" --> StepDone["Record Result and Mark COMPLETED"]
    StepDone --> ReleaseSem["Release Semaphore and Update Dependency Graph"]
    ReleaseSem --> TopoCheck

    Gate1Fail --> RetryCheck{"Attempt <= max_step_retries?"}
    ScoreEval -- "Failed" --> RetryCheck
    
    RetryCheck -- "Yes" --> BuildFeedback["Format retry_feedback from Reasoning and Missing Criteria"]
    BuildFeedback --> RunStep
    
    RetryCheck -- "No" --> StepFail["Mark Step FAILED with error_message"]
    StepFail --> ReleaseSem
```

---

## 5. Core Domain Data Model Relationships

```mermaid
classDiagram
    class Plan {
        +str goal
        +List~PlanStep~ steps
        +int version
        +int parent_version
        +str created_by
        +List~str~ assumptions
        +str change_summary
        +datetime created_at
    }

    class PlanStep {
        +str id
        +str description
        +str tool_name_hint
        +List~str~ depends_on
        +str success_criteria
        +StepStatus status
        +str result_summary
        +int attempts
        +str error_message
    }

    class RunBudget {
        +int max_total_tokens
        +float max_total_cost_usd
        +float total_timeout_seconds
        +int max_concurrency
        +int max_planning_retries
        +int max_step_retries
        +int max_replans
        +int current_tokens
        +float current_cost_usd
        +start_timer() void
        +record_usage(int tokens, float cost_usd) void
        +is_exhausted() bool
        +check_budget() void
    }

    class RunRecord {
        +str run_id
        +str goal
        +str user_id
        +RunStatus status
        +str final_deliverable
        +List~Plan~ plan_versions
        +Dict step_executions
        +int total_tokens
        +float total_cost_usd
        +float duration_ms
        +datetime created_at
        +datetime completed_at
    }

    class StepExecutionRecord {
        +str step_id
        +int plan_version
        +StepStatus status
        +int attempts
        +List~str~ depends_on
        +int tokens_used
        +float cost_usd
        +float duration_ms
        +str result_summary
        +StepVerificationResult verification
    }

    class StepVerificationResult {
        +str step_id
        +bool passed
        +bool programmatic_passed
        +bool model_passed
        +float score
        +str reasoning
        +List~str~ missing_criteria
    }

    Plan "1" *-- "*" PlanStep : contains
    RunRecord "1" *-- "*" Plan : plan_versions
    RunRecord "1" *-- "*" StepExecutionRecord : step_executions
    StepExecutionRecord "1" o-- "1" StepVerificationResult : verification
```
