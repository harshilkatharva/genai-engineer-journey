CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS long_term_memories (
    id uuid PRIMARY KEY,
    user_id text NOT NULL,
    content text NOT NULL CHECK (length(content) BETWEEN 1 AND 4000),
    category text NOT NULL CHECK (
        category IN ('fact', 'preference', 'goal', 'constraint', 'other')
    ),
    supersedes_ids uuid[] NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz,
    embedding vector(384) NOT NULL
);

CREATE INDEX IF NOT EXISTS long_term_memories_user_created_idx
    ON long_term_memories (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS long_term_memories_embedding_hnsw_idx
    ON long_term_memories USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS episodic_memories (
    id uuid PRIMARY KEY,
    user_id text NOT NULL,
    task_id text NOT NULL,  
    conversation_id text,
    action_type text NOT NULL,
    target text NOT NULL,
    parameters jsonb NOT NULL DEFAULT '{}',
    action_fingerprint text NOT NULL,
    status text NOT NULL CHECK (status IN ('attempted', 'completed', 'failed')),
    occurred_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL
);

CREATE INDEX IF NOT EXISTS episodic_memories_action_idx
    ON episodic_memories (user_id, task_id, action_fingerprint, occurred_at DESC);
CREATE INDEX IF NOT EXISTS episodic_memories_expiry_idx
    ON episodic_memories (expires_at);
