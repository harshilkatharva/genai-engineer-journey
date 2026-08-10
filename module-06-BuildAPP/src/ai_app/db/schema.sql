CREATE EXTENSION IF NOT EXISTS pgcrypto;


CREATE TABLE conversations (
    conversation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    user_id VARCHAR(255) NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


CREATE TABLE history (
    id BIGSERIAL PRIMARY KEY,

    conversation_id UUID NOT NULL
        REFERENCES conversations(conversation_id)
        ON DELETE CASCADE,

    user_id VARCHAR(255) NOT NULL,

    request_id UUID NOT NULL,

    role VARCHAR(20) NOT NULL
        CHECK (
            role IN (
                'system',
                'user',
                'assistant',
                'tool'
            )
        ),

    content TEXT NOT NULL,

    feature VARCHAR(50) NOT NULL
        CHECK (
            feature IN (
                'chat',
                'summarization',
                'sentiment'
            )
        ),

    llm_model VARCHAR(100),

    message_tokens INTEGER NOT NULL DEFAULT 0,

    input_tokens INTEGER NOT NULL DEFAULT 0,

    output_tokens INTEGER NOT NULL DEFAULT 0,

    estimated_cost NUMERIC(12, 8) NOT NULL DEFAULT 0,

    duration_ms INTEGER,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


CREATE INDEX idx_history_conversation
ON history(conversation_id, created_at, id);

CREATE INDEX idx_history_request
ON history(request_id);

CREATE INDEX idx_history_user
ON history(user_id);