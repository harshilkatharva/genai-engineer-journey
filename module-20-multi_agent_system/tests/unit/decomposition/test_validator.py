"""Unit tests for DecompositionValidator."""

from __future__ import annotations

import pytest

from parallel_research_team.decomposition.validator import DecompositionValidator
from parallel_research_team.exceptions import DecompositionError
from parallel_research_team.models import (
    BriefType,
    EffortLevel,
    OutputFormatSpec,
    ResearchDecomposition,
    WorkerBrief,
)


def _make_brief(brief_id: str, title: str) -> WorkerBrief:
    return WorkerBrief(
        brief_id=brief_id,
        title=title,
        topic="Vector Databases",
        objective=f"Objective for {title}",
        brief_type=BriefType.COMPARATIVE_ANALYSIS,
        task_boundaries=[f"Focus on {title}", "Do NOT research other domains"],
        expected_output_format=OutputFormatSpec(),
    )


def test_validator_valid_direct_plan() -> None:
    validator = DecompositionValidator()
    brief = _make_brief("b1", "Single Topic")
    plan = ResearchDecomposition(
        topic="What is HNSW?",
        effort_level=EffortLevel.DIRECT,
        rationale="Narrow concept lookup",
        worker_count=1,
        worker_briefs=[brief],
    )
    # Should pass without error
    validator.validate(plan, max_workers=4)


def test_validator_valid_parallel_plan() -> None:
    validator = DecompositionValidator()
    b1 = _make_brief("b1", "HNSW Performance")
    b2 = _make_brief("b2", "DiskANN Scale")
    plan = ResearchDecomposition(
        topic="Compare HNSW and DiskANN",
        effort_level=EffortLevel.PARALLEL_DECOMPOSED,
        rationale="Comparative multi-system question",
        worker_count=2,
        worker_briefs=[b1, b2],
    )
    validator.validate(plan, max_workers=4)


def test_validator_empty_topic_or_rationale() -> None:
    validator = DecompositionValidator()
    b1 = _make_brief("b1", "Title")

    plan1 = ResearchDecomposition(
        topic="Valid",
        effort_level=EffortLevel.DIRECT,
        rationale="Valid",
        worker_count=1,
        worker_briefs=[b1],
    )
    # Modify topic to whitespace
    object.__setattr__(plan1, "topic", "   ")
    with pytest.raises(DecompositionError, match="topic cannot be empty"):
        validator.validate(plan1)

    plan2 = ResearchDecomposition(
        topic="Valid",
        effort_level=EffortLevel.DIRECT,
        rationale="Valid",
        worker_count=1,
        worker_briefs=[b1],
    )
    object.__setattr__(plan2, "rationale", "   ")
    with pytest.raises(DecompositionError, match="rationale cannot be empty"):
        validator.validate(plan2)


def test_validator_exceeding_max_workers() -> None:
    validator = DecompositionValidator()
    briefs = [_make_brief(f"b{i}", f"Title {i}") for i in range(5)]
    plan = ResearchDecomposition(
        topic="Broad Topic",
        effort_level=EffortLevel.PARALLEL_DECOMPOSED,
        rationale="Very broad",
        worker_count=5,
        worker_briefs=briefs,
    )
    with pytest.raises(DecompositionError, match="exceeding configured max_workers=3"):
        validator.validate(plan, max_workers=3)


def test_validator_direct_with_multiple_briefs_fails() -> None:
    validator = DecompositionValidator()
    b1 = _make_brief("b1", "Title 1")
    b2 = _make_brief("b2", "Title 2")
    plan = ResearchDecomposition(
        topic="Topic",
        effort_level=EffortLevel.DIRECT,
        rationale="Simple",
        worker_count=2,
        worker_briefs=[b1, b2],
    )
    with pytest.raises(DecompositionError, match="DIRECT effort level must produce exactly 1"):
        validator.validate(plan, max_workers=4)


def test_validator_parallel_with_single_brief_fails() -> None:
    validator = DecompositionValidator()
    b1 = _make_brief("b1", "Title 1")
    plan = ResearchDecomposition(
        topic="Topic",
        effort_level=EffortLevel.PARALLEL_DECOMPOSED,
        rationale="Decomposed",
        worker_count=1,
        worker_briefs=[b1],
    )
    with pytest.raises(
        DecompositionError, match="PARALLEL_DECOMPOSED effort level must produce at least 2"
    ):
        validator.validate(plan, max_workers=4)


def test_validator_effort_override_conflict() -> None:
    validator = DecompositionValidator()
    b1 = _make_brief("b1", "Title 1")
    plan = ResearchDecomposition(
        topic="Topic",
        effort_level=EffortLevel.DIRECT,
        rationale="Direct",
        worker_count=1,
        worker_briefs=[b1],
    )
    with pytest.raises(DecompositionError, match="conflicts with requested override"):
        validator.validate(plan, max_workers=4, effort_override=EffortLevel.PARALLEL_DECOMPOSED)


def test_validator_duplicate_brief_titles_fails() -> None:
    validator = DecompositionValidator()
    b1 = _make_brief("b1", "Identical Scope")
    b2 = _make_brief("b2", "Identical Scope")  # Duplicate title/scope
    plan = ResearchDecomposition(
        topic="Topic",
        effort_level=EffortLevel.PARALLEL_DECOMPOSED,
        rationale="Decomposed",
        worker_count=2,
        worker_briefs=[b1, b2],
    )
    with pytest.raises(DecompositionError, match="Duplicate brief title detected"):
        validator.validate(plan, max_workers=4)
