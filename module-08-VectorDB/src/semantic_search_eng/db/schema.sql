CREATE TABLE document_chunks(
    id BIGSERIAL PRIMARY KEY,
    tenant_id UUID NOT NULL,
    document_id UUID NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(1536) NOT NULL,
    document_type TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'
);


CREATE INDEX ON document_chunks (tenant_id);

CREATE INDEX ON document_chunks USING hnsw (embedding vector_cosine_ops);