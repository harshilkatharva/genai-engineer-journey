# Module 19: Self-Improving Report Writer — Implemented Execution & Data Flow

This document reflects the actual implemented execution and data flows in Module 19. It is created in Milestone 1 and updated incrementally after each completed milestone.

---

## 1. Milestone 1 Implemented Architecture

Milestone 1 establishes the core typed domain models and Gate 1 deterministic verification engine:

```mermaid
flowchart TD
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

    subgraph VerifierEngine["2. Gate 1 Deterministic Verification (src/self_improving_report_writer/verification/)"]
        InputReport(["Report Candidate"])
        SourcePool[("Application-Controlled\nSource Pool")]
        NormHelper["normalize_for_matching()\n(lowercase, whitespace collapse, quote normalization)"]
        Verifier["DeterministicVerifier.verify()"]

        CheckSchema["Check 1: Schema & Integrity Invariants\n(Unique section_ids, claim_ids, citation_ids)"]
        CheckAuthority["Check 2: Source Authority\n(Check source_id & chunk_id exist in pool)"]
        CheckQuotations["Check 3: Quotation Matching\n(Exact / Normalized Substring Containment)"]
        CheckClaims["Check 4: Factual Citation Presence\n(Factual claims must have citations)"]
        CheckProse["Check 5: Prose Reference Alignment\n(Inline [cite_id] markers in content)"]

        ReportOutput(["DeterministicVerificationReport\n(passed, issues, passage_matches)"])

        InputReport & SourcePool --> Verifier
        Verifier --> CheckSchema
        CheckSchema --> CheckAuthority
        CheckAuthority --> CheckQuotations
        CheckQuotations <--> NormHelper
        CheckQuotations --> CheckClaims
        CheckClaims --> CheckProse
        CheckProse --> ReportOutput
    end
```

---

## 2. Gate 1 Deterministic Verification Data Flow

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
