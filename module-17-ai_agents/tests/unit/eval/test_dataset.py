"""Unit tests for the 15 evaluation benchmark goal definitions."""

from __future__ import annotations

from research_and_support_agent.eval.dataset import (
    create_15_goals_dataset,
    get_15_benchmark_samples,
)


def test_15_benchmark_samples_count_and_uniqueness() -> None:
    """Verify exactly 15 benchmark goal definitions exist with unique identifiers."""
    samples = get_15_benchmark_samples()
    assert len(samples) == 15

    sample_ids = [s.id for s in samples]
    assert len(sample_ids) == len(set(sample_ids)), "Duplicate sample ID detected"


def test_all_samples_have_queries_and_categories() -> None:
    """Verify all benchmark samples contain queries, categories, and expected tools."""
    samples = get_15_benchmark_samples()
    for s in samples:
        assert s.query, f"Sample {s.id} has empty query"
        assert s.metadata and "category" in s.metadata, f"Sample {s.id} missing category metadata"
        assert s.expected_tools, f"Sample {s.id} has no expected tools specified"


def test_create_benchmark_dataset_builder() -> None:
    """Verify EvalDataset wrapper construction."""
    ds = create_15_goals_dataset()
    assert ds.metadata.get("name") == "module17_15_goals"
    assert len(ds.samples) == 15
