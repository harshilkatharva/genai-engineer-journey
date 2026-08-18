import asyncio
import time
import asyncpg
import pytest

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request


# 1

query = """
BEGIN;

-- 1. Enable pgvector.
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Add the column as nullable so existing rows remain valid
--    while the backfill runs.
ALTER TABLE documents
ADD COLUMN embedding vector(1536);

-- 3. Backfill existing rows.
UPDATE documents
SET embedding = compute_embedding(content)
WHERE embedding IS NULL
  AND content IS NOT NULL;

-- 4. Enforce the invariant once the backfill is complete.
ALTER TABLE documents
ALTER COLUMN embedding SET NOT NULL;

-- 5. Optional: add an index for similarity search.
--    HNSW is generally a good default for pgvector.
CREATE INDEX documents_embedding_hnsw_idx
ON documents
USING hnsw (embedding vector_cosine_ops);

COMMIT;
"""


# 2


async def create_db_pool():
    return await asyncpg.create_pool(
        "postgresql://user:password@localhost/mydb",
        min_size=2,
        max_size=10,
    )


async def get_result(pool, user_id: int, query_embedding: list[float]):
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            """
            SELECT id, name
            FROM users
            WHERE id = $1
            ORDER BY embedding <=> $2
            """,
            user_id,
            query_embedding,
        )


# 3


def pct(xs, p):
    xs = sorted(xs)
    return xs[int(len(xs) * p)]


async def benchmark(pool, sizes, samples=100):
    query = "[" + ",".join(["0.01"] * 1536) + "]"  # real query vector

    for size in sizes:
        times = []

        async with pool.acquire() as conn:
            for _ in range(samples):
                start = time.perf_counter()

                await conn.fetch(
                    """
                    SELECT id
                    FROM documents
                    WHERE id <= $1
                    ORDER BY embedding <=> $2::vector
                    LIMIT 10
                    """,
                    size,
                    query,
                )

                times.append((time.perf_counter() - start) * 1000)

        print(
            f"{size:>10,} rows | "
            f"p50 {pct(times, 0.50):.2f}ms | "
            f"p95 {pct(times, 0.95):.2f}ms | "
            f"p99 {pct(times, 0.99):.2f}ms"
        )


async def run_benchmark():
    pool = await asyncpg.create_pool(
        "postgresql://user:password@localhost/mydb"  # real db connection url
    )
    try:
        await benchmark(
            pool,
            [10_000, 100_000, 1_000_000, 5_000_000],
        )
    finally:
        await pool.close()


# 4
async def upsert_chunks(
    pool: asyncpg.Pool, document_id: int, chunks: list[tuple[int, str]], embeddings: list[float]
):
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM document_chunks WHERE document_id = $1",
                document_id,
            )

            await conn.executemany(
                """
                INSERT INTO document_chunks
                    (document_id, chunk_index, content, embedding)
                VALUES ($1, $2, $3, $4::vector)
                """,
                [
                    (
                        document_id,
                        chunk.index,
                        chunk.content,
                        "[" + ",".join(map(str, embedding)) + "]",
                    )
                    for chunk, embedding in zip(chunks, embeddings)
                ],
            )


# 5
async def validate_embedding_dimension(
    pool: asyncpg.Pool,
    embedding: list[float],
    table: str = "document_chunks",
    column: str = "embedding",
) -> None:
    async with pool.acquire() as conn:
        dimension = await conn.fetchval(
            """
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = $1::regclass
              AND attname = $2
              AND NOT attisdropped
            """,
            table,
            column,
        )

    # pgvector stores VECTOR(N)'s dimension in atttypmod.
    if dimension is None:
        raise ValueError(f"{table}.{column} does not exist")

    if len(embedding) != dimension:
        raise ValueError(
            f"Embedding dimension mismatch: "
            f"query has {len(embedding)} dimensions, "
            f"but {table}.{column} expects {dimension}."
        )


# 6
query = """
WITH results AS (
    SELECT
        id,
        content,

        ts_rank(search_vector, websearch_to_tsquery('english', $1))
            AS text_score,

        1 - (embedding <=> $2::vector)
            AS vector_score

    FROM documents

    WHERE search_vector @@ websearch_to_tsquery('english', $1)
       OR embedding IS NOT NULL
)

SELECT
    id,
    content,
    text_score,
    vector_score,

    0.4 * text_score +
    0.6 * vector_score AS combined_score

FROM results

ORDER BY combined_score DESC

LIMIT $3;
"""


# 7
async def report_hnsw(
    pool,
    table="documents",
    index="documents_embedding_hnsw_idx",
    rows=1_000_000,
    m=16,
    ef_construction=64,
    dimensions=1536,
):
    async with pool.acquire() as conn:
        size = await conn.fetchval(
            "SELECT pg_size_pretty(pg_relation_size($1::regclass))",
            index,
        )

    # Rough HNSW estimate: ~M * 8 bytes per neighbor + vector storage.
    vector_bytes = rows * dimensions * 4
    graph_bytes = rows * m * 8
    estimated = vector_bytes + graph_bytes

    print(f"Index: {index}")
    print(f"Corpus: {rows:,} rows")
    print(f"M: {m}")
    print(f"ef_construction: {ef_construction}")
    print(f"Actual index size: {size}")
    print(f"Estimated memory: {estimated / 1024**3:.2f} GB")


async def run_report():
    pool = await asyncpg.create_pool("postgresql://postgres:postgres@localhost:5432/ai_search")
    try:
        await report_hnsw(
            pool,
            rows=1_000_000,
            m=16,
            ef_construction=64,
            dimensions=1536,
        )
    finally:
        await pool.close()


asyncio.run(run_report())


# 8
@pytest.mark.asyncio
async def test_tenant_isolation():
    # 1. Seed data as admin
    admin_pool = await asyncpg.create_pool(
        "postgresql://postgres:postgres@localhost:5432/ai_search"
    )

    async with admin_pool.acquire() as conn:
        await conn.execute("DELETE FROM documents_tenant_depth")

        await conn.executemany(
            """
            INSERT INTO documents_tenant_depth
                (tenant_id, content, embedding)
            VALUES ($1, $2, $3)
            """,
            [
                (1, "Tenant A secret", "[1,0,0]"),
                (2, "Tenant B secret", "[1,0,0]"),
            ],
        )

    await admin_pool.close()

    # 2. Query as normal application user
    pool = await asyncpg.create_pool("postgresql://app_user:app_password@localhost:5432/ai_search")

    async with pool.acquire() as conn:
        # Pretend this request belongs to tenant A
        await conn.execute("SET app.tenant_id = '1'")

        # INTENTIONALLY NO tenant_id WHERE CLAUSE!
        rows = await conn.fetch(
            """
            SELECT tenant_id, content
            FROM documents_tenant_depth
            """
        )

    await pool.close()

    # 3. Tenant B must never appear
    assert rows
    assert all(row["tenant_id"] == 1 for row in rows)
    assert all("Tenant B" not in row["content"] for row in rows)


# 9
DATABASE_URL = "postgresql://app_user:app_password@localhost:5432/ai_search"

# Example: expect ~20 concurrent DB queries.
MIN_CONNECTIONS = 5
MAX_CONNECTIONS = 20


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=MIN_CONNECTIONS,
        max_size=MAX_CONNECTIONS,
        command_timeout=10,
    )

    yield

    await app.state.db.close()


app = FastAPI(lifespan=lifespan)


@app.get("/search")
async def search(request: Request, q: str):
    pool: asyncpg.Pool = request.app.state.db

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                id,
                content,
                1 - (embedding <=> $1::vector) AS similarity
            FROM documents
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1::vector
            LIMIT 10
            """,
            "[0.01,0.02,0.03]",
        )

    return [dict(row) for row in rows]


# 10
async def monitor_index(pool):
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT
                pg_size_pretty(
                    pg_relation_size('document_chunks_embedding_idx')
                ) AS index_size,
                pg_size_pretty(
                    pg_relation_size('document_chunks')
                ) AS table_size,
                pg_size_pretty(
                    pg_total_relation_size('document_chunks')
                ) AS total_size
        """)

        print(
            f"Table: {row['table_size']} | Index: {row['index_size']} | Total: {row['total_size']}"
        )
