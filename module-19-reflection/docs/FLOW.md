# Module 19: Self-Improving Report Writer — Implemented Execution & Data Flow

This document reflects the actual implemented execution and data flows in Module 19. It is created in Milestone 1 and updated incrementally after each completed milestone.

---

## 1. Implemented Architecture (Milestones 1, 2 & 3)

```mermaid
flowchart TB
    subgraph DataStructures["1. Domain Models (src/self_improving_report_writer/models/)"]
        SourceRef["SourceReference\n(Immutable application source chunk)"]
        CitationModel["Citation\n(QUOTATION | PARAPHRASE)"]
        ClaimModel["Claim\n(Factual proposition & attached citations)"]
        SectionModel["ReportSection\n(Section prose & structured claims)"]
        ReportModel["Report\n(Executive Summary, Sections, Conclusion, Limitations)"]
        RubricModel["RubricConfig & DimensionScore\n(7 calibrated dimensions, sum=1.0)"]

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

    %% Wiring
    TopicInput(["Research Topic"]) --> Retriever
    Retriever -->|list of SourceReference| PromptBuilder
    PromptBuilder --> Generator
    Generator -->|Report Candidate| G1Verifier
    Retriever -->|Authoritative Source Pool| G1Verifier
    Retriever -->|Authoritative Source Pool| G2Verifier
    G1Verifier -->|DeterministicReport| G2Verifier
    G2Verifier --> SemanticOutput(["SemanticGroundingReport"])
    Generator --> Judge
    G1Verifier --> Judge
    Retriever --> Judge
    Judge --> VerdictOutput(["ReportJudgeVerdict"])
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
    Judge->>Judge: Check dimension min_pass_score across all 7 dimensions
    Judge-->>Caller: ReportJudgeVerdict(composite_score, passed, dimension_scores, feedback)
    end
```

