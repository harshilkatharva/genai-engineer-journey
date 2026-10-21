# Module 17: Bounded Research and Support Agent — Execution & Data Flow

## 1. End-to-End Execution Flow

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Client
    participant Agent as BoundedAgent
    participant Memory as MemoryService (Postgres)
    participant Runner as ToolLoopRunner (genai-core)
    participant Executor as ToolExecutor (genai-core)
    participant HITL as PostgresApprovalStore
    actor Human as Operator / Approver

    User->>Agent: Send Query (e.g., "Refund customer C-101 $50")
    
    rect rgb(240, 245, 255)
    Note over Agent,Memory: Phase 1: Context Preparation & Recall
    Agent->>Memory: recall(user_id, query="refund customer tone constraints")
    Memory-->>Agent: [MemorySearchResult(content="Customer is Gold Tier", score=0.95)]
    Agent->>Agent: Construct System Prompt + Inject Recalled Long-Term Facts
    end

    rect rgb(245, 240, 255)
    Note over Agent,Runner: Phase 2: Bounded Tool Loop
    Agent->>Runner: run(messages, budget=ToolBudget, max_loop_repeats=3)
    
    Runner->>Executor: execute(lookup_customer_account(customer_id="C-101"))
    Executor-->>Runner: ToolResult(status=SUCCESS, content={tier: "Gold", balance: 50.0})

    Runner->>Executor: execute(execute_refund(customer_id="C-101", amount_usd=50.0))
    Note over Executor: RiskLevel.IRREVERSIBLE detected!
    Executor->>HITL: save_request(ApprovalRequest)
    Executor-->>Runner: (approved=False, ApprovalRequest)
    Runner-->>Agent: ToolLoopResult(status=APPROVAL_REQUIRED, pending_approval)
    end

    rect rgb(255, 245, 235)
    Note over Agent,Human: Phase 3: Durable Human-in-the-Loop Gating
    Agent->>User: "Paused: Refund requires support-team authorization."
    Human->>HITL: get_request(request_id)
    Human->>HITL: record_decision(request_id, ApprovalDecision(approved=True, modified_arguments={"amount_usd": 40.0}))
    end

    rect rgb(240, 255, 245)
    Note over Agent,Runner: Phase 4: Resume & Episodic Recording
    Agent->>Runner: resume(decision, request, messages, budget)
    Runner->>Executor: execute(execute_refund, bypass_approval=True)
    Executor->>Memory: check_action("execute_refund", {"customer_id": "C-101", "amount_usd": 40.0})
    Memory-->>Executor: ActionCheck(status="execute")
    Executor->>Executor: Execute refund operation
    Executor->>Memory: record_action("execute_refund", status="success")
    Executor-->>Runner: ToolResult(status=SUCCESS, content="Refund of $40.00 processed.")
    Runner-->>Agent: ToolLoopResult(status=COMPLETED, final_text="Processed $40.00 refund.")
    end

    Agent->>Agent: TranscriptRecorder writes transcripts/run_<id>.json
    Agent-->>User: Final Response + Telemetry Summary
```

## 2. Component Interactions

1. **`BoundedAgent`**:
   - Primary facade coordinating initialization, pre-turn long-term memory recall, budgeting, `ToolLoopRunner` execution, and post-run transcript serialization.
2. **`ToolLoopRunner` (`genai-core`)**:
   - Manages the multi-turn interaction loop with `LLMService`.
   - Checks `ToolBudget.can_step()` before each iteration.
   - Enforces monotonic deadlines and cumulative token/cost ceilings.
   - Intercepts consecutive identical tool invocations via `LoopDetector`.
3. **`ToolExecutor` (`genai-core`)**:
   - Manages asynchronous tool dispatching with concurrency limits (`asyncio.Semaphore`).
   - Enforces per-tool execution timeouts.
   - Gates actions matching `require_approval_for={RiskLevel.IRREVERSIBLE}`.
4. **`PostgresApprovalStore`**:
   - Persists pending approval requests in table `approval_requests`.
   - Persists reviewer decisions in table `approval_decisions`.
   - Allows safe restart and resumption of interrupted execution sessions.
5. **`MemoryService` (`genai-core`) backed by `PostgresMemoryStore`**:
   - Manages durable storage in tables `memory_lt_records` and `memory_ep_episodes`.
   - Computes vector embeddings and recency decay for long-term recall.
   - Computes canonical SHA-256 parameter hashes for episodic action deduplication.
6. **`TranscriptRecorder`**:
   - Collects per-step duration (latency ms), input/output tokens, and USD cost.
   - Emits standardized JSON transcripts to `transcripts/` directory.
