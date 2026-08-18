# ADR: Choose pgvector for the Production Vector Store

**Date:** 2026-08-18  
**Decision Type:** Infrastructure / Data Architecture

## 1. Context

We are building a multi-tenant knowledge-search service for an AI application.

The system will initially handle approximately:

- **5 million document chunks**
- **Up to 100 tenants**
- **Up to 100 semantic-search queries/second at peak**
- **Target p95 retrieval latency:** below **100 ms**
- **Metadata filtering:** required by `tenant_id` and `document_type`
- **PostgreSQL:** assuming we are already used by the application for users, documents, permissions, and billing
- **Team:** 2-4 engineers, with limited operational capacity; the team does not want to operate another production database unless there is a demonstrated need

The vector store must support approximate nearest-neighbor search, metadata filtering, tenant isolation, batch ingestion, and incremental updates.

## 2. Options Considered

### Option A — PostgreSQL + pgvector

Advantages:

- Reuses the existing PostgreSQL infrastructure.
- Supports vector storage and similarity search directly in PostgreSQL.
- Allows normal SQL filtering and joins alongside vector search.
- Keeps vectors and application data in the same transactional system.
- Avoids introducing another system to deploy, secure, monitor, and maintain.
- Supports HNSW and IVFFlat indexing.

Trade-offs:

- Dedicated vector databases can provide more specialized vector-search functionality and potentially a higher scaling ceiling.
- Specialized ANN tuning and vector-specific capabilities may be less extensive than in dedicated products.

### Option B — Dedicated Vector Database

Examples include Pinecone, Weaviate, and Qdrant.

Advantages:

- Purpose-built for vector workloads.
- Designed for very large-scale vector search.
- Can provide more specialized vector tuning and scaling capabilities.

Trade-offs:

- Introduces another production system.
- Adds operational, security, monitoring, and cost overhead.
- Application data and vector data become separate systems.
- Cross-system consistency becomes more complicated.

## 3. Decision

**We will use PostgreSQL + pgvector.**

The decision is based on the constraints above.

The expected corpus of **5 million chunks** is substantial but still moderate relative to the scenarios where a dedicated vector system becomes necessary. The application already depends on PostgreSQL, and if the team has limited operational capacity.

The pgvector architecture can provide:

```text
Application
```

↓

Query Service

↓

PostgreSQL + pgvector

├── B-tree indexes → tenant/document filtering

└── HNSW index    → vector similarity search

This gives us both:

```text
semantic search
+
structured metadata filtering
```

within the same database.

The module's comparison framework strongly favors pgvector when PostgreSQL is already present and scale is moderate because it avoids introducing another system while retaining expressive SQL filtering, transactions, and vector search.

## 4. Why HNSW

We will initially use **HNSW** rather than an unindexed exact search.

Exact search performs a linear scan over the vector collection and therefore becomes increasingly expensive as the corpus grows.

HNSW provides approximate nearest-neighbor search with a speed/recall trade-off that is appropriate for our latency target.

We will benchmark its parameters rather than treating them as fixed defaults.

## 5. Multi-Tenant Design

Tenant isolation is a correctness and security requirement.

The schema will contain:

```text
    tenant_id,
    document_id,
    chunk_id,
    chunk_text,
    embedding,
    document_type,
    metadata
```

Queries will enforce:

```sql
WHERE tenant_id = ...
```

at the database level rather than retrieving all vectors and filtering them in application code.

A B-tree index will support common metadata filtering, while HNSW will support vector similarity search.

## 6. What Would Change This Recommendation?

We would reconsider pgvector and evaluate a dedicated vector database if one or more of these conditions became true:

### 1. Scale increases substantially

For example, the corpus grows from approximately **5 million** chunks to hundreds of millions or beyond, and PostgreSQL becomes a demonstrated bottleneck.

### 2. Query volume becomes much larger

If peak traffic grows from roughly **100 QPS** to several thousand or tens of thousands of vector queries per second and pgvector cannot meet the required latency.

### 3. Specialized vector features become necessary

For example, we require vector-specific capabilities or ANN tuning that pgvector cannot provide adequately.

### 4. Vector workload needs operational isolation

If vector search becomes large enough that its CPU, memory, or storage workload interferes significantly with the application's transactional PostgreSQL workload.

### 5. Benchmarking proves pgvector cannot meet requirements

The most important trigger is empirical evidence.

If our production-shaped benchmark shows:

```text
Recall       below target
p95 latency  above target
QPS          below target
```

even after reasonable indexing and tuning, we would revisit the architecture.

## 7. Consequences

### Positive

- No additional database platform to operate initially.
- Existing PostgreSQL expertise can be reused.
- Vector and application data can participate in the same database transactions.
- Rich SQL metadata filtering remains available.
- Simpler architecture and deployment.

### Negative

- We accept that a dedicated vector database may eventually provide a higher scaling ceiling or more specialized capabilities.
- PostgreSQL now carries both transactional and vector-search workloads.
- We must benchmark HNSW configuration and monitor database resource usage.

## 8. Final Decision Rule

Our decision is therefore:

> **Start with pgvector, measure continuously, and migrate only when a demonstrated workload requirement—not hype or assumption—justifies a dedicated vector database.**

This follows the core principle from Module 8:

```text
Existing PostgreSQL

        +

Moderate scale

        +

Strong SQL filtering needs

        +

Limited operational capacity

        ↓
     pgvector



```

A dedicated vector database becomes the better choice when:

```text
Extreme scale

OR

Very high vector QPS

OR

Specialized vector requirements

OR

Measured pgvector limitations
```