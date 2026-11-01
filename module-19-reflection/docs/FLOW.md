# Module 19: Self-Improving Report Writer — Implemented Execution & Data Flow

This document reflects the actual implemented execution and data flows in Module 19. It is created in Milestone 1 and updated incrementally after each completed milestone.

---

## 1. Implemented Architecture (Milestones 1 & 2)

```mermaid
flowchart TB
    subgraph DataStructures["1. Domain Models (src/self_improving_report_writer/models/)"]
        SourceRef["SourceReference\n(Immutable application source chunk)"]
        CitationModel["Citation\n(QUOTATION | PARAPHRASE)"]
        ClaimModel["Claim\n(Factual proposition & attached citations)"]
        SectionModel["ReportSection\n(Section prose & structured claims)"]
        ReportModel["Report\n(Executive Summary, Sections, Conclusion, Limitations)"]

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

    subgraph VerifierEngine["4. Gate 1 Deterministic Verification (src/self_improving_report_writer/verification/)"]
        Verifier["DeterministicVerifier.verify()"]
        CheckSchema["Check 1: Structural Integrity"]
        CheckAuthority["Check 2: Application Source Authority"]
        CheckQuotations["Check 3: Exact / Normalized Substring Match"]
        CheckClaims["Check 4: Factual Citation Presence"]
        CheckProse["Check 5: Prose Citation Alignment"]

        Verifier --> CheckSchema --> CheckAuthority --> CheckQuotations --> CheckClaims --> CheckProse
    end

    %% Wiring
    TopicInput(["Research Topic"]) --> Retriever
    Retriever -->|list of SourceReference| PromptBuilder
    PromptBuilder --> Generator
    Generator -->|Report Candidate| Verifier
    Retriever -->|Authoritative Source Pool| Verifier
    Verifier --> ReportOutput(["DeterministicVerificationReport"])
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
