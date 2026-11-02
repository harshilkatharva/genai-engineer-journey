# Module 19: Self-Improving Report Writer — Implemented Execution & Data Flow

This document reflects the actual implemented execution and data flows in Module 19. It is created in Milestone 1 and updated incrementally after each completed milestone.

---

## 1. Implemented Architecture (Milestones 1, 2, 3 & 4)

```mermaid
flowchart TB
    subgraph DataStructures["1. Domain Models (src/self_improving_report_writer/models/)"]
        SourceRef["SourceReference\n(Immutable application source chunk)"]
        CitationModel["Citation\n(QUOTATION | PARAPHRASE)"]
        ClaimModel["Claim\n(Factual proposition & attached citations)"]
        SectionModel["ReportSection\n(Section prose & structured claims)"]
        ReportModel["Report\n(Executive Summary, Sections, Conclusion, Limitations)"]
        RubricModel["RubricConfig & DimensionScore\n(7 calibrated dimensions, sum=1.0)"]
        CritiqueModel["Critique & CritiqueIssue\n(Prioritized actionable defects, should_revise)"]

        SourceRef -.-> CitationModel
        CitationModel --> ClaimModel
        ClaimModel --> SectionModel
        SectionModel --> ReportModel
    end

    subgraph RetrievalLayer["2. Evidence Retrieval (src/self_improving_report_writer/retrieval/)"]
        Retriever["SourceRetriever\n(Facade over genai-core RetrievalService)"]
        StoreAdapter[("VectorStore / InMemoryStore\n(768-dim embeddings)")]
        Retriever <--> StoreAdapter
    end

    subgraph GenerationLayer["3. Report Synthesis (src/self_improving_report_writer/generation/)"]
        PromptBuilder["build_report_generation_prompt()\n(Untrusted XML delimiter barrier)"]
        Generator["ReportGenerator\n(LLMService complete_structure)"]
        PromptBuilder --> Generator
    end

    subgraph VerifierEngine["4. Verification Pipeline (src/self_improving_report_writer/verification/)"]
        G1Verifier["Gate 1: DeterministicVerifier.verify()\n(Pure Python, 0 LLM calls)"]
        G2Verifier["Gate 2: SemanticGroundingVerifier.verify_grounding()\n(Claim NLI Entailment & Prose Consistency)"]

        G1Verifier -->|If Pass| G2Verifier
    end

    subgraph JudgingEngine["5. Rubric Evaluation (src/self_improving_report_writer/judging/)"]
        Judge["RubricJudge.evaluate()\n(gemini-2.5-pro / diversity fallback)"]
        Rollup["Deterministic Math Rollup\n(Weighted composite score >= threshold)"]
        Judge --> Rollup
    end

    subgraph ReflectionEngine["6. Reflection & Feedback (src/self_improving_report_writer/reflection/)"]
        CritiqueEng["CritiqueEngine.generate_critique()\n(Translates defects, enforces revision thresholds)"]
        OscillationDet["OscillationDetector.check_oscillation()\n(SHA-256 Content & Critique Fingerprinting)"]
        CritiqueEng --> OscillationDet
    end

    %% Wiring
    TopicInput(["Research Topic"]) --> Retriever
    Retriever -->|list of SourceReference| PromptBuilder
    PromptBuilder --> Generator
    Generator -->|Report Candidate| G1Verifier
    Retriever -->|Authoritative Source Pool| G1Verifier
    Retriever -->|Authoritative Source Pool| G2Verifier
    G1Verifier -->|DeterministicReport| G2Verifier
    Generator --> Judge
    G1Verifier --> Judge
    Retriever --> Judge
    G1Verifier --> CritiqueEng
    G2Verifier --> CritiqueEng
    Judge --> CritiqueEng
    CritiqueEng -->|Critique| OscillationDet
    OscillationDet --> LoopDecision(["OscillationCheckResult\n(content_cycle | critique_stagnation | none)"])
```



---

## 2. Milestone 2: Evidence Retrieval & Synthesis Data Flow

```mermaid
sequenceDiagram
    autonumber
    actor Caller as Caller / Orchestrator
    participant Ret as SourceRetriever
    participant CoreRet as genai-core.RetrievalService
    participant PB as PromptBuilder (prompts.py)
    participant Gen as ReportGenerator
    participant LLM as genai-core.LLMService
    participant Ver as DeterministicVerifier

    Caller->>Ret: retrieve_sources(topic="Quantum Annealing", top_k=8, min_score=0.0)
    Ret->>CoreRet: search(query="Quantum Annealing", top_k=8)
    CoreRet-->>Ret: RetrievalResponse(results=[RetrievalResult, ...])
    Ret-->>Caller: list[SourceReference] (immutable source chunks)

    Caller->>Gen: generate_report(topic, sources, guidelines)
    Gen->>PB: build_report_generation_prompt(topic, sources, guidelines)
    Note over PB: Enclose chunks in <untrusted_source_context> tags
    Note over PB: Sanitize internal delimiter occurrences
    PB-->>Gen: complete prompt string with security barrier

    Gen->>LLM: complete_structure(prompt, response_schema=Report)
    LLM-->>Gen: LLMManagerResponse(structured_response=Report, usage)
    Gen-->>Caller: (Report, tokens_used, cost_usd)

    Caller->>Ver: verify(report, retrieved_sources=sources)
    Ver-->>Caller: DeterministicVerificationReport(passed, issues)
```

---

## 3. Gate 1 Deterministic Verification Logic

```mermaid
sequenceDiagram
    autonumber
    actor Caller as Caller / Test Suite
    participant Ver as DeterministicVerifier
    participant Report as Report Instance
    participant Pool as Application Source Pool

    Caller->>Ver: verify(report, retrieved_sources)
    Ver->>Ver: _build_source_map(retrieved_sources)
    
    rect rgb(240, 245, 255)
    Note over Ver,Report: Stage 1: Structural & Integrity Invariants
    Ver->>Report: Check mandatory sections (exec_summary, sections >= 1, conclusion)
    Ver->>Report: Verify claim.section_id == section.section_id
    Ver->>Report: Verify claim text non-empty
    end

    rect rgb(245, 240, 255)
    Note over Ver,Pool: Stage 2: Application Source Authority Check
    loop For each Citation in each Claim
        Ver->>Pool: lookup (citation.source_id, citation.chunk_id)
        alt Not found in Pool
            Ver->>Ver: Record CRITICAL defect (Unknown source/chunk pair)
            Ver->>Ver: Set citations_valid = False
        else Found in Pool
            Note over Ver: Source reference verified
        end
    end
    end

    rect rgb(255, 250, 240)
    Note over Ver,Pool: Stage 3: Exact / Normalized Quotation Matching
    loop For each Citation where citation_type == QUOTATION
        Ver->>Ver: normalize_for_matching(citation.quoted_passage)
        Ver->>Ver: normalize_for_matching(source_chunk.content)
        alt Exact substring match
            Ver->>Ver: Record PassageMatchResult(is_match=True, match_type='exact')
        else Normalized substring match (incl. enclosing quote strip)
            Ver->>Ver: Record PassageMatchResult(is_match=True, match_type='normalized')
        else Substring Missing
            Ver->>Ver: Record PassageMatchResult(is_match=False, match_type='missing')
            Ver->>Ver: Record CRITICAL defect (Quotation missing from chunk)
            Ver->>Ver: Set passages_matched = False
        end
    end
    end

    rect rgb(240, 255, 245)
    Note over Ver,Report: Stage 4: Prose Citation Alignment
    Ver->>Report: Check inline [cite_id] markers in section.content
    alt Citation defined but not referenced in prose
        Ver->>Ver: Record MINOR defect (Unreferenced citation marker)
    end
    end

    Ver-->>Caller: DeterministicVerificationReport(passed, issues, counts)
```

---

## 4. Gate 2 Semantic Grounding & Rubric Judging Data Flow

```mermaid
sequenceDiagram
    autonumber
    actor Caller as Caller / Orchestrator
    participant G2 as SemanticGroundingVerifier
    participant Judge as RubricJudge
    participant LLM as genai-core.LLMService
    participant Report as Report Candidate
    participant Sources as Authoritative Source Pool

    rect rgb(240, 248, 255)
    Note over Caller,G2: Gate 2: Semantic Grounding Verification
    Caller->>G2: verify_grounding(report, retrieved_sources)
    loop For each factual Claim
        alt No citations attached
            G2->>G2: Flag MAJOR defect: Uncited Factual Claim (Score 0.0)
        else Citations reference missing chunk
            G2->>G2: Flag MAJOR defect: Missing Chunk ID (Score 0.0)
        else Citations resolved
            G2->>LLM: complete_structure(claim_text, cited_chunks, response_schema=ClaimEntailmentEvaluation)
            LLM-->>G2: ClaimEntailmentEvaluation(entailment, score, reasoning)
            alt ENTAILS
                G2->>G2: Score 1.0 (Supported)
            else NEUTRAL
                G2->>G2: Flag MAJOR defect: Weak/Neutral Grounding (Score 0.5)
            else CONTRADICTS
                G2->>G2: Flag CRITICAL defect: Evidence Contradiction (Score 0.0)
            end
        end
    end
    loop For each section prose
        G2->>LLM: complete_structure(prose, structured_claims, response_schema=ProseConsistencyEvaluation)
        LLM-->>G2: ProseConsistencyEvaluation(unstructured_claims)
        opt Unstructured claims present
            G2->>G2: Flag MAJOR defect: Ungrounded Factual Prose Assertion
        end
    end
    G2-->>Caller: SemanticGroundingReport(passed, grounding_score, issues)
    end

    rect rgb(255, 248, 240)
    Note over Caller,Judge: Rubric-Based Evaluation & Scoring
    Caller->>Judge: evaluate(report, sources, deterministic_report)
    Judge->>Judge: Check model diversity (gemini-2.5-pro vs generator model)
    opt Model Diversity Compromised (Self-Judging)
        Judge->>Judge: Set is_self_judged=True, Threshold bumped +0.05 (0.85 -> 0.90)
    end
    Judge->>LLM: complete_structure(prompt, response_schema=LLMJudgeEvaluation, provider=judge_model)
    opt Primary Judge Fails
        Judge->>LLM: Fallback to generator_model, set is_self_judged=True, threshold +0.05
    end
    LLM-->>Judge: LLMJudgeEvaluation(dimension_scores, overall_feedback)
    Judge->>Judge: Pure Python deterministic composite score calculation (sum w_i * score_i)
    Judge-->>Caller: ReportJudgeVerdict(composite_score, passed, dimension_scores, feedback)
    end
```

---

## 5. Structured Critique Generation & Oscillation Detection Flow

```mermaid
sequenceDiagram
    autonumber
    actor Caller as Reflection Orchestrator
    participant CE as CritiqueEngine
    participant OD as OscillationDetector
    participant Det as DeterministicVerificationReport (G1)
    participant Grd as SemanticGroundingReport (G2)
    participant Jdg as ReportJudgeVerdict (Rubric)
    participant Rep as Report Candidate

    Caller->>CE: generate_critique(round, report, det_report, grd_report, judge_verdict)
    
    rect rgb(245, 240, 255)
    Note over CE: Defect Translation & Categorization
    CE->>Det: Ingest schema, authority, and substring quotation defects
    CE->>Grd: Ingest entailment (CONTRADICTS, NEUTRAL) and ungrounded prose defects
    CE->>Jdg: Ingest failing rubric dimensions (< min_pass_score or < 0.80)
    CE->>CE: Deduplicate issues by signature and sort (CRITICAL > MAJOR > MINOR)
    end

    rect rgb(240, 255, 245)
    Note over CE: Severity-Based Revision Threshold Decision
    alt critical_count > 0
        CE->>CE: should_revise = True (Critical issues always revise)
    else major_count > 0
        CE->>CE: should_revise = (composite_score < threshold OR grounding_score < 0.85)
    else minor_count > 0 AND critical=0 AND major=0
        opt composite_score >= threshold
            CE->>CE: should_revise = False (Immediate Acceptance, terminate loop)
        end
    end
    CE-->>Caller: Critique(round_number, issues, summary, should_revise)
    end

    rect rgb(255, 250, 240)
    Note over Caller,OD: Dual-State Fingerprinting & Oscillation Check
    Caller->>OD: check_oscillation(round_number, report, critique)
    OD->>OD: compute_content_fingerprint(report) -> SHA256(norm_title + norm_sections)
    OD->>OD: compute_critique_fingerprint(critique) -> SHA256(sorted({category+claim+severity}))
    alt content_hash in content_history
        OD-->>Caller: OscillationCheckResult(has_oscillation=True, type='content_cycle')
        Note over Caller: Loop terminated: report reverted to earlier version
    else critique_hash == critique_history[-1]
        OD-->>Caller: OscillationCheckResult(has_oscillation=True, type='critique_stagnation')
        Note over Caller: Loop terminated: revision failed to fix identified defects
    else No oscillation detected
        OD->>OD: Record content_hash and critique_hash into history
        OD-->>Caller: OscillationCheckResult(has_oscillation=False, type='none')
    end
    end
```

---

## 6. Milestone 5: Bounded Reflection Orchestration & Candidate Selection Flow

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Benchmark Runner
    participant Orch as ReflectionLoopOrchestrator
    participant Budget as RunBudget (Shared Multi-Resource Ceiling)
    participant Ret as SourceRetriever
    participant Gen as ReportGenerator
    participant Ver as Dual-Gate (G1 Verifier & G2 Grounding)
    participant Jdg as RubricJudge
    participant CE as CritiqueEngine
    participant OD as OscillationDetector
    participant Rev as RevisionEngine
    participant Sel as CandidateSelector (Hierarchical Pareto)

    User->>Orch: run(topic, sources=None, initial_report=None)
    Orch->>Budget: Initialize RunBudget(max_tokens=50k, max_cost=$0.20, max_time=90s, max_rounds=3)
    Orch->>OD: reset()

    rect rgb(240, 245, 255)
    Note over Orch,Gen: Phase 1: Source Resolution & Baseline Generation (Round 0)
    Orch->>Ret: retrieve_sources(topic)
    Ret-->>Orch: source_list (immutable SourceReference chunks)
    Orch->>Gen: generate_report(topic, sources)
    Gen-->>Orch: (Report v1, tokens, cost)
    Orch->>Budget: record_usage(tokens, cost)
    end

    rect rgb(245, 240, 255)
    Note over Orch,Jdg: Phase 2: Baseline Candidate Evaluation
    Orch->>Ver: G1: verify(report, sources) & G2: verify_grounding(report, sources)
    Ver-->>Orch: DeterministicVerificationReport + SemanticGroundingReport
    Orch->>Budget: record_usage(g2_tokens, g2_cost)
    Orch->>Jdg: evaluate(report, sources, det_report)
    Jdg-->>Orch: ReportJudgeVerdict(composite_score, passed)
    Orch->>Budget: record_usage(judge_tokens, judge_cost)
    Orch->>Orch: candidates.append(ReportCandidate(round=0, ...))
    Orch->>CE: generate_critique(round=0, report, det, g2, judge)
    CE-->>Orch: Critique(round=0, issues, should_revise)
    opt not critique.should_revise
        Orch-->>User: ReflectionRunResult(best=Round 0, termination="ALL_ACCEPTANCE_CRITERIA_MET")
    end
    Orch->>OD: check_oscillation(round=0, report, critique)
    end

    rect rgb(255, 248, 240)
    Note over Orch,Rev: Phase 3: Bounded Iterative Revision Loop (Rounds 1..max_rounds)
    loop Round k = 1 to max_rounds (default: 3)
        Orch->>Budget: check_budget_exceeded(current_round=k-1)
        opt Budget Exceeded (Tokens, Cost, or Time)
            Note over Orch: Break loop with BUDGET_EXCEEDED termination
        end

        Orch->>Rev: revise_report(current_report, sources, critique)
        Rev-->>Orch: (Report v{k+1}, rev_tokens, rev_cost)
        Orch->>Budget: record_usage(rev_tokens, rev_cost)

        Orch->>Ver: G1 verify() + G2 verify_grounding()
        Ver-->>Orch: det_report, g2_report
        Orch->>Budget: record_usage(g2_tokens, g2_cost)
        Orch->>Jdg: evaluate(revised_report, sources, det_report)
        Jdg-->>Orch: judge_verdict
        Orch->>Budget: record_usage(judge_tokens, judge_cost)

        Orch->>Orch: rev_cand = ReportCandidate(round=k, ...)
        Orch->>Sel: is_regression(candidates[-1], rev_cand)
        alt Regression detected
            Orch->>Orch: consecutive_regressions += 1
            opt consecutive_regressions >= max_consecutive_regressions (1)
                Note over Orch: Break loop with CONSECUTIVE_REGRESSIONS_EXCEEDED
            end
        else Quality preserved or improved
            Orch->>Orch: consecutive_regressions = 0
        end

        Orch->>Orch: candidates.append(rev_cand)
        Orch->>Budget: record_round_completed()

        Orch->>CE: generate_critique(round=k, rev_cand...)
        CE-->>Orch: Critique(round=k, issues, should_revise)
        opt not critique.should_revise
            Note over Orch: Break loop with ALL_ACCEPTANCE_CRITERIA_MET
        end

        Orch->>OD: check_oscillation(round=k, revised_report, critique)
        opt has_oscillation (content_cycle or critique_stagnation)
            Note over Orch: Break loop with OSCILLATION_DETECTED
        end

        Orch->>Budget: check_budget_exceeded(current_round=k)
        opt Budget ceiling reached
            Note over Orch: Break loop with BUDGET_EXCEEDED
        end
    end
    end

    rect rgb(240, 255, 245)
    Note over Orch,Sel: Phase 4: Deterministic Hierarchical Candidate Selection
    Orch->>Sel: select_best(candidates)
    Note over Sel: Tier 1: Gate 1 Pass > Fail<br/>Tier 2: Fewer Critical Defects<br/>Tier 3: Fewer Major Defects<br/>Tier 4: All Claims Grounded & Grounding Score<br/>Tier 5: Composite Rubric Score<br/>Tier 6: Grounding Sub-Scores<br/>Tier 7: Parsimony (Earlier round)
    Sel-->>Orch: best_candidate (guaranteed optimal, immune to late regressions)
    Orch-->>User: ReflectionRunResult(best_candidate, all_candidates, budget_summary, termination_reason)
    end
```

---

## 7. Milestone 6: Acceptance Gate & Human Review Gateway Flow

```mermaid
sequenceDiagram
    autonumber
    actor Caller as Caller / User
    participant Gate as AcceptanceGate
    participant GW as HumanReviewGateway
    actor Operator as Human Operator
    participant Rev as RevisionEngine
    participant Ver as Dual-Gate & Rubric Judge

    Caller->>Gate: evaluate_run_result(run_result)
    Note over Gate: Evaluate 5 Mandatory Acceptance Criteria:<br/>1. Gate 1 deterministic passed == True<br/>2. Critical defect count == 0<br/>3. Major factual grounding defects == 0<br/>4. Composite score >= threshold (0.85 / 0.90 self-judged)<br/>5. Factual Accuracy & Evidence Grounding >= 0.85

    alt All 5 Conditions Met
        Gate-->>Caller: AcceptanceResult(status=ACCEPTED, is_accepted=True, review_packet=None)
        Note over Caller: Report deliverable finalized and ready for consumption
    else Any Condition Failed
        Gate->>Gate: Construct HumanReviewPacket(defects, scores, critiques)
        Gate-->>Caller: AcceptanceResult(status=UNRESOLVED, is_accepted=False, review_packet=packet)
        Caller->>GW: format_review_packet(packet)
        GW-->>Operator: Display markdown inspection packet

        Operator->>GW: process_decision(HumanReviewDecision)
        alt Action: APPROVE_OVERRIDE
            GW-->>Caller: HumanReviewOutcome(status=HUMAN_APPROVED, notes preserved, verifications intact)
        else Action: REJECT
            GW-->>Caller: HumanReviewOutcome(status=HUMAN_REJECTED)
        else Action: REQUEST_REVISION (Round <= 2)
            GW->>Rev: revise_report(best_candidate, sources, critique with operator directives)
            Rev-->>GW: revised_report (version=v+1, created_by='human_revision')
            GW->>Ver: Re-verify Gate 1, Gate 2, and RubricJudge
            Ver-->>GW: revised_candidate
            GW->>Gate: evaluate_candidate(revised_candidate)
            Gate-->>GW: AcceptanceResult(status=ACCEPTED or UNRESOLVED)
            GW-->>Caller: HumanReviewOutcome(status=result.status, final_report, revised_candidate)
        end
    end
```

---

## 8. Milestone 7: Episodic Memory Lesson Distillation & Retrieval Flow

```mermaid
sequenceDiagram
    autonumber
    actor Orch as Reflection Orchestrator / Pipeline
    participant MM as ReflectionMemoryManager
    participant EM as genai-core.EpisodicMemory
    participant LLM as genai-core.LLMService

    rect rgb(240, 248, 255)
    Note over Orch,EM: Phase A: Pre-Generation Memory Retrieval
    Orch->>MM: retrieve_lessons(user_id, domain, limit=5)
    MM->>EM: get_episodes(user_id, action_type="reflection_writing_lesson", limit=20)
    EM-->>MM: list[Episode]
    MM->>MM: Filter episodes where metadata['domain'] == domain
    MM-->>Orch: list[str] (distilled abstract guidelines)
    Note over Orch: Injected into ReportGenerator system prompt
    end

    rect rgb(255, 248, 240)
    Note over Orch,EM: Phase B: Post-Run Lesson Distillation
    Orch->>MM: distill_lessons(user_id, domain, initial_cand, best_cand, session_id)
    MM->>MM: Verify improvement: delta_score >= 0.10 OR delta_critical >= 1
    alt Improvement Insufficient
        MM-->>Orch: [] (Lesson recording skipped)
    else Verified Improvement Satisfied
        MM->>LLM: Distill resolved issues into abstract rules (confidence, rule, category)
        LLM-->>MM: list[DistilledLessonCandidate]
        loop For each candidate lesson with confidence >= 0.85
            MM->>EM: check(user_id, action_type="reflection_writing_lesson", parameters, target)
            alt Duplicate exists (check() == True)
                MM->>MM: Skip duplicate
            else Unique lesson
                MM->>EM: record(user_id, action_type="reflection_writing_lesson", parameters, target, result_summary=rule)
                MM->>MM: Append to recorded lessons
            end
        end
        MM-->>Orch: list[str] (newly recorded lessons)
    end
    end
```

---

## 9. Milestone 8: 15-Topic Comparative Benchmark Evaluation Flow

```mermaid
sequenceDiagram
    autonumber
    actor CLI as Benchmark CLI / Operator
    participant Runner as BenchmarkRunner
    participant Dataset as 15-Topic Benchmark Dataset
    participant AppA as Approach A Runner (No Reflection)
    participant AppB as Approach B Runner (Ungrounded Refinement)
    participant AppC as Approach C Runner (Grounded Reflection)
    participant Judge as BlindedJudge (Unbiased Rubric)
    participant Gate as AcceptanceGate
    participant Metrics as Metrics Calculator
    participant Out as Markdown & JSON Output

    CLI->>Runner: run_benchmark(topic_ids=all, output_dir="benchmark_outputs")
    Runner->>Dataset: get_benchmark_dataset()
    Dataset-->>Runner: 15 Benchmark Topics (5 Domains x 3 Topics)

    loop For each Benchmark Topic (1 to 15)
        rect rgb(240, 248, 255)
        Note over Runner,AppA: Condition A: No Reflection (Baseline)
        Runner->>AppA: run_approach_a(topic)
        AppA->>AppA: Generate draft R0
        AppA->>Judge: evaluate_blinded(R0)
        AppA->>Gate: evaluate_candidate(candidate_A)
        AppA-->>Runner: ApproachRunResult(rounds=0, A_scores, A_defects, A_cost)
        end

        rect rgb(255, 250, 240)
        Note over Runner,AppB: Condition B: Ungrounded Self-Critique
        Runner->>AppB: run_approach_b(topic, max_rounds=2)
        AppB->>AppB: Generate draft R0
        loop Round 1 to 2
            AppB->>AppB: Free-form LLM Self-Critique (no verifier access)
            AppB->>AppB: Revise draft to R_k
        end
        AppB->>Judge: evaluate_blinded(R_final)
        AppB->>Gate: evaluate_candidate(candidate_B)
        AppB-->>Runner: ApproachRunResult(rounds=k, B_scores, B_defects, B_cost)
        end

        rect rgb(240, 255, 245)
        Note over Runner,AppC: Condition C: Grounded Reflection (Our System)
        Runner->>AppC: run_approach_c(topic)
        AppC->>AppC: ReflectionLoopOrchestrator(Dual Gates + Structured Critique + Oscillation Detector + Pareto Selection)
        AppC->>Judge: evaluate_blinded(best_candidate.report)
        AppC->>Gate: evaluate_candidate(best_candidate)
        AppC-->>Runner: ApproachRunResult(rounds=k, C_scores, C_defects, C_cost)
        end
    end

    rect rgb(250, 240, 255)
    Note over Runner,Out: Aggregation & Deliverable Generation
    Runner->>Metrics: compute_comparative_summary(all_results)
    Metrics->>Metrics: Calculate means, std, pass rates, cost deltas, and research question resolution
    Metrics-->>Runner: ComparativeEvaluationSummary
    Runner->>Out: Save benchmark_summary.md (Markdown table & analysis)
    Runner->>Out: Save benchmark_results.json (Machine-readable audit trail)
    Runner-->>CLI: BenchmarkExecutionReport
    end
```

