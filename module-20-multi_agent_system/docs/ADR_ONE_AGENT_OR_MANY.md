# Architecture Decision Record (ADR): One Agent or Many?

- **Document ID:** ADR-20-001
- **Status:** Decided (Recommendation: Default to Single Agent; Restrict Multi-Agent to Complex Multi-Faceted Surveys via Dynamic Routing)
- **Date:** 2026-11-25
- **Scope:** Technical Research & Intelligence Synthesis Architecture (`module-17` Bounded Single Agent vs. `module-20` Parallel Research Team)
- **Source of Truth References:**
  - Evaluation Benchmark Run: `eval_results/eval_run_eval_63515ddcd6.json` (15 questions across 4 archetypes)
  - Runtime Checkpoints: `checkpoints/checkpoint_run_*.json` (49 persisted checkpoint runs)
  - Core Configuration: `src/parallel_research_team/core/config.py`
  - Lifecycle Architecture: `docs/FLOW.md`
  - System Decisions: `docs/DECISIONS.md` (DEC-20-029, DEC-20-030, DEC-20-031)

---

## 1. Context and Problem Statement

When building automated research and synthesis systems over heterogeneous information sources (combining live web search via Tavily and internal vector knowledge bases via pgvector/in-memory embeddings), engineering teams face a fundamental architectural choice:

1. **One Agent:** A single bounded agent running a sequential ReAct loop with a single tool registry and scratchpad.
2. **Many Agents:** A multi-agent system decomposing the inquiry into parallel sub-problems, dispatching independent specialized workers, aggregating intermediate findings, and executing structured verification gates.

The goal of this Architecture Decision Record (ADR) is to evaluate whether the operational complexity, financial cost, and latency overhead of a multi-agent system are justified by measurable quality improvements, or whether a single agent is the superior production choice.

---

## 2. The Task Description

The target workload is **Comprehensive Technical Research & Intelligence Reporting**. Given a user query, the system must:
- Retrieve authoritative information across web and internal vector sources.
- Synthesize facts into a structured report (`Report` schema with executive summary, dedicated technical sections, conclusions, and explicit limitations).
- Ground claims with granular citations (`[cite_N]`) mapped to concrete source chunk IDs (`SourceReference`).
- Pass deterministic structural validation (Gate 1) and semantic entailment grounding verification (Gate 2).

---

## 3. Analysis of the Three Single-Agent Ceilings

Single-agent systems typically fail when they collide with one or more of three architectural ceilings:

### Ceiling 1: Context Window Ceiling (Information Density & Distraction)
- **Observed Behavior:**
  - For **Simple Factual** (`q01`–`q04`) and **Linear Synthesis** (`q05`–`q07`) queries, the single agent consumes an average of only **3,596 to 4,111 tokens**. Context window limits are never challenged, and information density remains high.
  - For **Broad Architectural Surveys** (`q13`, `q14`), the single agent retrieves limited chunks (typically 3–5 items) to stay within its prompt budget, causing it to produce narrower syntheses (scoring **4.00/5.00**).
  - Conversely, the multi-agent system distributes literature discovery across parallel workers, each maintaining an isolated scratchpad. In queries requiring orthogonal domain analysis (e.g., `q13_tesla_niagara_impact` and `q14_agentic_architectures_tradeoffs`), the multi-agent system achieved **5.00/5.00** quality by overcoming the single agent's context saturation ceiling.

### Ceiling 2: Tool Namespace Ceiling (Tool Complexity & Conflict)
- **Observed Behavior:**
  - The research workload requires exactly two read-only retrieval tools: `search_web` and `search_knowledge_base`.
  - In our benchmarks, the single agent handles both tools without namespace collisions, parameter confusion, or schema degradation.
  - **Empirical Finding:** This task **does NOT hit the tool namespace ceiling**. Introducing multi-agent delegation solely for tool routing creates unnecessary orchestration overhead without providing any namespace isolation benefit.

### Ceiling 3: Sequential Work Ceiling (Wall-Clock Latency & Breadth of Search)
- **Observed Behavior:**
  - A single agent executes retrieval sequentially: tool call $\rightarrow$ tool response $\rightarrow$ reflection $\rightarrow$ next tool call.
  - For queries that naturally branch into independent sub-domains (e.g., comparing four distinct RAG metrics in `q10` or architectural trade-offs in `q14`), a single agent must investigate sequentially.
  - However, in practice, the multi-agent system's orchestration pipeline (decomposition prompt + worker dispatch + worker execution + synthesis prompt + Gate 1 + Gate 2 + reflection loop) introduces substantial serialization delays. As measured, multi-agent runs averaged **160.43s** compared to the single agent's **15.81s** (a **10.1x latency penalty**).

---

## 4. Multi-Agent Pattern Selection: Hierarchical Orchestrator-Workers

Where multi-agent execution was implemented (`module-20`), the pattern chosen was **Hierarchical Orchestrator-Workers with Map-Reduce Decomposition and Two-Gate Synthesis**:

```
                       User Research Topic
                               │
                    ┌──────────▼──────────┐
                    │ DecompositionEngine │
                    │   (Lead / Planner)  │
                    └──────────┬──────────┘
                               │
               ┌───────────────┼───────────────┐
        WorkerBrief 1   WorkerBrief 2   WorkerBrief N (<=4)
               │               │               │
        ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
        │ WorkerTask  │ │ WorkerTask  │ │ WorkerTask  │
        │  (Isolated) │ │  (Isolated) │ │  (Isolated) │
        └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
               │               │               │
         WorkerResult    WorkerResult    WorkerResult
               └───────────────┼───────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   SynthesisEngine   │
                    │    (Aggregator)     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Gate 1 Deterministic│
                    │ Gate 2 Semantic NLI │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  ReflectionEngine   │
                    │ (Self-Correction)   │
                    └──────────┬──────────┘
                               │
                         Final Report
```

### Rationale for this Pattern
1. **Isolated Worker Contexts:** Prevents cross-worker prompt pollution. Each worker receives a strict `WorkerBrief` and executes in an isolated environment with a dedicated step limit ($\le 5$ steps) and timeout ($45.0$s).
2. **Deterministic Checkpointing:** State is persisted atomically at every lifecycle boundary (`FileCheckpointStore`), allowing runs to resume cleanly after system interruptions.
3. **Budget Guardrails:** A thread-safe `SharedRunBudget` tracks token and financial spend against hard global limits ($60,000$ tokens, $\$0.50$ USD).
4. **Autonomous Self-Correction:** When Gate 2 identifies weak or neutral claim entailment, `ReflectionEngine` repairs specific claims against authoritative source chunks within an isolated budget envelope ($15,000$ tokens, $\$0.05$ USD).

---

## 5. Agent Contracts and Permissions

To maintain isolation and security, permissions are strictly segregated:

| Agent / Component | Input Contract | Output Contract | Granted Permissions | Prohibited Actions |
|---|---|---|---|---|
| **Decomposition Engine** | User `topic`, `effort_level` | `DecompositionPlan` ($\le 4$ `WorkerBrief` objects) | LLM completion (`complete_structure`), Checkpoint Store read/write | Web search, tool execution, direct report writing |
| **Research Worker (`WorkerTask`)** | `WorkerBrief` (query, focus, required sources, schema) | `WorkerResult` (findings, structured sections, raw sources) | Read-only access to `search_web` and `search_knowledge_base`, local scratchpad | Direct peer communication, checkpoint writes, final report compilation |
| **Synthesis Engine** | Aggregated `WorkerResult` list, original `topic` | Candidate `Report` object | LLM completion (`complete_structure`), source deduplication | Direct tool execution, external network calls |
| **Gate Verifiers (1 & 2)** | Candidate `Report`, authoritative source chunks | Verification status, claim entailment scores, issue list | Deterministic regex parser, LLM judge (`judge_model`) | Mutating report content, executing search tools |
| **Reflection Engine** | Candidate `Report`, Gate 2 defect list, source chunks | Revised `Report` object | Bounded LLM completion ($\le 15\text{k}$ tokens, $\le \$0.05$, $\le 2$ steps) | Expanding research scope, invoking search tools |

---

## 6. Empirical Evaluation Evidence

The comparative evaluation was conducted using the benchmark pipeline across 15 questions categorized into 4 distinct archetypes. All evaluations ran against identical live search and vector retrieval backends.

*Primary Artifact: `eval_results/eval_run_eval_63515ddcd6.json`*

### 6.1 Macro Performance Summary

| Metric | Single Agent (`module-17`) | Multi-Agent (`module-20`) | Delta / Multiplier |
|---|---|---|---|
| **Wins** | **13 / 15 (86.7%)** | **2 / 15 (13.3%)** | Single Agent +11 wins |
| **Average Quality Score** | **4.87 / 5.00** | **4.44 / 5.00** | Single Agent +0.43 score |
| **Average Wall-Clock Latency** | **15.81 seconds** | **160.43 seconds** | Multi-Agent **10.1x slower** |
| **Average Cost per Query** | **$0.000530 USD** | **$0.019959 USD** | Multi-Agent **37.7x more expensive** |
| **Average Tokens per Query** | **4,040 tokens** | **30,361 tokens** | Multi-Agent **7.5x more tokens** |

---

### 6.2 Archetype Breakdown and Recorded Checkpoint Evidence

```
Archetype Win Distribution:
  Simple Factual (4/4):           [SA] [SA] [SA] [SA]
  Linear Synthesis (3/3):         [SA] [SA] [SA]
  Multi-Faceted Comparison (4/4): [SA] [SA] [SA] [SA]
  Architectural Survey (2/4):     [SA] [SA] [MA] [MA]
```

#### Archetype 1: Simple Factual Queries (`q01`–`q04`)
- **Queries:** `q01_clarissa_recipient`, `q02_ivfflat_concept`, `q03_tesla_patent_year`, `q04_minilm_dimension`
- **Result:** Single Agent won **4 / 4**.
- **Metrics:**
  - Latency: Single Agent **15.28s** vs. Multi-Agent **185.02s** (12.1x slower)
  - Cost: Single Agent **$0.000472** vs. Multi-Agent **$0.007609** (16.1x higher)
  - Quality: Single Agent **5.00** vs. Multi-Agent **4.71**
- **Checkpoint Evidence (`checkpoints/checkpoint_run_0451e2b348c8.json`):**
  - For query `q01` (*"Who is the primary recipient of Clarissa Harlowe's letters in Volume 1?"*), the multi-agent system spent **17,014 tokens**, **$0.010446**, and **166.37 seconds** to produce an over-engineered multi-section report with redundant executive summaries and limitations, while the single agent returned the exact correct answer in **18.16 seconds** for **$0.001226**.

#### Archetype 2: Linear Synthesis Queries (`q05`–`q07`)
- **Queries:** `q05_pricing_calculation`, `q06_clarissa_family_motive`, `q07_hnsw_skiplist_heuristic`
- **Result:** Single Agent won **3 / 3**.
- **Metrics:**
  - Latency: Single Agent **16.92s** vs. Multi-Agent **88.72s** (5.2x slower)
  - Cost: Single Agent **$0.000539** vs. Multi-Agent **$0.017564** (32.6x higher)
  - Quality: Single Agent **5.00** vs. Multi-Agent **3.60**
- **Failure Analysis:** In `q05_pricing_calculation`, decomposition split a sequential calculation across workers, resulting in fragmented arithmetic where the multi-agent system scored **1.80/5.00** on quality, whereas the single agent scored **5.00/5.00** in **6.53 seconds** at **$0.000081**.

#### Archetype 3: Multi-Faceted Comparisons (`q08`–`q11`)
- **Queries:** `q08_hnsw_vs_ivfflat`, `q09_tesla_sync_vs_torque`, `q10_rag_eval_metrics`, `q11_pgvector_vs_dedicated_vdb`
- **Result:** Single Agent won **4 / 4** on efficiency tie-breaking.
- **Metrics:**
  - Latency: Single Agent **19.81s** vs. Multi-Agent **176.76s** (8.9x slower)
  - Cost: Single Agent **$0.000813** vs. Multi-Agent **$0.024912** (30.7x higher)
  - Quality: Single Agent **5.00** vs. Multi-Agent **4.75**
- **Checkpoint Evidence (`checkpoints/checkpoint_run_0c4c7b16829e.json`):**
  - For `q10_rag_eval_metrics`, multi-agent allocated 2 workers (`brief_01`, `brief_02`) consuming **48,806 tokens**, **$0.027386**, and **437.81 seconds**. The single agent achieved higher quality (5.00 vs 4.00) in **28.03 seconds** at **$0.001576**.

#### Archetype 4: Complex Architectural Surveys (`q12`–`q15`)
- **Queries:** `q12_production_genai_scope`, `q13_tesla_niagara_impact`, `q14_agentic_architectures_tradeoffs`, `q15_enterprise_llm_cache_guardrails`
- **Result:** Single Agent won 2 (`q12`, `q15`), Multi-Agent won 2 (`q13`, `q14`).
- **Metrics:**
  - Quality: Single Agent **4.50** vs. Multi-Agent **4.50**
  - Latency: Single Agent **11.50s** vs. Multi-Agent **173.29s**
  - Cost: Single Agent **$0.000299** vs. Multi-Agent **$0.029151**
- **Where Multi-Agent Won Decisively:**
  - **`q13_tesla_niagara_impact`:** Multi-Agent Quality **5.00** vs. Single Agent Quality **4.00** (+1.00 quality advantage).
  - **`q14_agentic_architectures_tradeoffs`:** Multi-Agent Quality **5.00** vs. Single Agent Quality **4.00** (+1.00 quality advantage).
  - On these broad inquiries, multi-worker parallel literature exploration captured architectural trade-offs and historical impact points that the single agent truncated due to sequential context limits.

---

## 7. Decision and Recommendation

Based strictly on empirical evaluation checkpoints and recorded system performance:

### Primary Decision: Default to Single Agent
1. **The Single Agent is the recommended default architecture** for the vast majority of research and retrieval tasks.
2. In 86.7% of evaluated scenarios (13 out of 15 questions), the single agent delivered equivalent or superior quality at **10.1x lower latency** and **37.7x lower cost**.
3. Deploying a multi-agent system indiscriminately incurs severe performance degradation, unnecessary token spend, and higher operational failure surface (e.g., handling worker timeouts, decomposition errors, and synthesis retries).

### Secondary Decision: Conditional Routing for Deep Architectural Surveys
Multi-agent execution is justified **only** under strict qualifying criteria:
- The inquiry requires parallel investigation across $\ge 2$ mutually orthogonal domains.
- The expected synthesis exceeds the single-agent prompt context envelope ($\ge 15\text{k}$ raw retrieved tokens).
- The user or workflow explicitly requests an exhaustive multi-section intelligence report and accepts a $>120$-second latency profile and a $\sim 30\text{x}$ cost multiplier.

### Implementation Rule: Architectural Intent Gateway
Future production deployments should place an **Intent Classifier Gateway** ahead of execution:
- If query intent is `factual`, `linear_calculation`, or `focused_comparison` $\rightarrow$ route directly to `SingleAgentAdapter` (`module-17`).
- If query intent is `broad_architectural_survey` or `exhaustive_multi_domain_investigation` $\rightarrow$ route to `ParallelResearchTeam` (`module-20`).

---

## 8. Summary Scorecard

```
Architecture Comparison Summary:
┌──────────────────────────────┬────────────────────────┬────────────────────────┐
│ Dimension                    │ Single Agent (Mod 17)  │ Multi-Agent (Mod 20)   │
├──────────────────────────────┼────────────────────────┼────────────────────────┤
│ Architecture Style           │ Sequential ReAct Loop  │ Hierarchical M/R Team  │
│ Benchmark Wins (n=15)        │ 13 (86.7%)             │ 2 (13.3%)              │
│ Average Overall Quality      │ 4.87 / 5.00            │ 4.44 / 5.00            │
│ Average Latency              │ 15.81s                 │ 160.43s (10.1x slower) │
│ Average Cost per Run         │ $0.000530              │ $0.019959 (37.7x more) │
│ Failure Recovery Complexity  │ Low (Single loop)      │ High (Atomic CKPT/Gate)│
│ Best Fit                     │ 90% of Queries         │ Deep Multi-Angle Study │
└──────────────────────────────┴────────────────────────┴────────────────────────┘
```
