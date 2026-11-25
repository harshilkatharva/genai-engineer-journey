"""Unit tests for AuthoritativeSourcePool."""

from __future__ import annotations

import concurrent.futures

from parallel_research_team.models import SourceReference
from parallel_research_team.synthesis.source_pool import AuthoritativeSourcePool


def test_source_pool_add_and_get() -> None:
    pool = AuthoritativeSourcePool()
    assert len(pool) == 0

    src = SourceReference(
        source_id="web_abc123",
        chunk_id="chk_456",
        title="Web Page Title",
        content="Research text content.",
        score=0.95,
        metadata={"url": "https://example.com"},
    )
    pool.add_source(src)

    assert len(pool) == 1
    assert pool.has_source("web_abc123", "chk_456")
    assert pool.has_chunk("chk_456")

    retrieved = pool.get("web_abc123", "chk_456")
    assert retrieved is not None
    assert retrieved.title == "Web Page Title"

    by_chunk = pool.get_by_chunk_id("chk_456")
    assert by_chunk == retrieved


def test_source_pool_add_from_dict() -> None:
    pool = AuthoritativeSourcePool()
    raw = {
        "source_id": "kb_doc1",
        "chunk_id": "chunk_01",
        "title": "KB Policy",
        "content": "Company dispatch policy.",
        "score": 0.88,
    }
    pool.add_source(raw)
    assert len(pool) == 1
    assert pool.has_source("kb_doc1", "chunk_01")


def test_source_pool_deduplication() -> None:
    pool = AuthoritativeSourcePool()
    src1 = SourceReference(
        source_id="src_1",
        chunk_id="chk_1",
        title="Doc 1",
        content="Version A",
    )
    src2 = SourceReference(
        source_id="src_1",
        chunk_id="chk_1",
        title="Doc 1 Updated",
        content="Version B",
    )
    pool.add_source(src1)
    pool.add_source(src2)
    assert len(pool) == 1
    assert pool.get("src_1", "chk_1").content == "Version B"  # type: ignore[union-attr]


def test_source_pool_serialization_roundtrip() -> None:
    pool = AuthoritativeSourcePool()
    src = SourceReference(
        source_id="src_10",
        chunk_id="chk_10",
        title="Doc 10",
        content="Content 10",
    )
    pool.add_source(src)

    data = pool.to_dict()
    assert "src_10#chk_10" in data

    restored = AuthoritativeSourcePool.from_dict(data)
    assert len(restored) == 1
    assert restored.has_source("src_10", "chk_10")
    assert restored.get("src_10", "chk_10").content == "Content 10"  # type: ignore[union-attr]


def test_source_pool_preserves_colliding_chunk_ids() -> None:
    pool = AuthoritativeSourcePool()
    src1 = SourceReference(
        source_id="doc_1",
        chunk_id="chunk_0",
        title="Doc 1",
        content="Content 1",
    )
    src2 = SourceReference(
        source_id="doc_2",
        chunk_id="chunk_0",
        title="Doc 2",
        content="Content 2",
    )
    pool.add_source(src1)
    pool.add_source(src2)

    assert len(pool) == 2
    data = pool.to_dict()
    assert len(data) == 2
    assert "doc_1#chunk_0" in data
    assert "doc_2#chunk_0" in data

    restored = AuthoritativeSourcePool.from_dict(data)
    assert len(restored) == 2
    assert restored.has_source("doc_1", "chunk_0")
    assert restored.has_source("doc_2", "chunk_0")
    assert restored.get("doc_1", "chunk_0").content == "Content 1"  # type: ignore[union-attr]
    assert restored.get("doc_2", "chunk_0").content == "Content 2"  # type: ignore[union-attr]


def test_source_pool_thread_safety() -> None:
    pool = AuthoritativeSourcePool()

    def worker(i: int) -> None:
        for j in range(20):
            pool.add_source(
                SourceReference(
                    source_id=f"src_{i}",
                    chunk_id=f"chk_{i}_{j}",
                    title=f"Doc {i}-{j}",
                    content="Concurrent content",
                )
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker, i) for i in range(5)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    assert len(pool) == 100
