-- document_chunks is populated by the existing Modules 8-9 ingestion flow.
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT,
    status TEXT NOT NULL,
    ordered_at DATE,
    total_amount NUMERIC(12, 2),
    items JSONB NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    price NUMERIC(12, 2),
    available BOOLEAN NOT NULL DEFAULT TRUE,
    metadata JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS support_escalations (
    ticket_id BIGSERIAL PRIMARY KEY,
    customer_id TEXT,
    reason TEXT NOT NULL,
    summary TEXT NOT NULL,
    priority TEXT NOT NULL DEFAULT 'normal',
    status TEXT NOT NULL DEFAULT 'open',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
