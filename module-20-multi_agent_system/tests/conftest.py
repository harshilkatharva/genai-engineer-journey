"""Shared test fixtures for module-20-multi_agent_system test suite."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from parallel_research_team.core.config import ResearchTeamConfig
from parallel_research_team.models.brief import BriefType, OutputFormatSpec, WorkerBrief
from parallel_research_team.models.report import SourceReference

# Ensure src is in python path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def check_integration_gate(
    keywords: Mapping[str, Any] | set[str], integration_env: str | None
) -> None:
    """Skip test if it is marked with integration and INTEGRATION_TEST != '1'."""
    if "integration" in keywords and integration_env != "1":
        pytest.skip(
            "Integration tests are disabled. Set INTEGRATION_TEST=1 in .env or environment to run them."
        )


@pytest.fixture(autouse=True)
def require_integration_test_flag(request: pytest.FixtureRequest) -> None:
    """Gate tests marked with @pytest.mark.integration behind INTEGRATION_TEST=1."""
    check_integration_gate(request.keywords, os.getenv("INTEGRATION_TEST"))


@pytest.fixture
def test_config(tmp_path: Path) -> ResearchTeamConfig:
    """Provides isolated ResearchTeamConfig pointing to a temporary checkpoint directory."""
    return ResearchTeamConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        max_total_tokens=10_000,
        max_total_cost_usd=0.10,
        max_wall_clock_seconds=30.0,
        max_concurrency=3,
        max_workers=3,
        per_worker_timeout_seconds=5.0,
        per_worker_max_steps=3,
        judge_provider="test",
        judge_model="test-model-1",
        semantic_verification_threshold=0.70,
    )


@pytest.fixture
def sample_sources() -> list[SourceReference]:
    """Sample immutable SourceReference items for testing."""
    return [
        SourceReference(
            source_id="src_vector_01",
            chunk_id="chk_vec_01",
            title="Vector Indexing Strategies",
            content="HNSW graphs achieve lower latency than IVF at the expense of higher index build time.",
            score=0.92,
            metadata={"domain": "databases", "year": 2026},
        ),
        SourceReference(
            source_id="src_vector_02",
            chunk_id="chk_vec_02",
            title="Graph-based Vector Indexing",
            content="DiskANN enables billion-scale vector search on a single server with SSD storage.",
            score=0.88,
            metadata={"domain": "databases", "year": 2026},
        ),
    ]


@pytest.fixture
def sample_brief() -> WorkerBrief:
    """Sample valid WorkerBrief for testing."""
    return WorkerBrief(
        brief_id="brief_01",
        title="HNSW vs DiskANN Trade-offs",
        topic="Vector Indexing Architectures",
        objective="Investigate latency and memory trade-offs between HNSW and DiskANN",
        brief_type=BriefType.COMPARATIVE_ANALYSIS,
        task_boundaries=[
            "Focus strictly on indexing latency and memory overhead",
            "Do NOT investigate quantization or clustering",
        ],
        expected_output_format=OutputFormatSpec(
            schema_name="WorkerResearchOutput",
            required_fields=["findings_summary", "claims", "sources"],
            guidelines="Structured findings with claims and source citations",
        ),
        suggested_tools=["search_web", "search_knowledge_base"],
        max_steps=5,
        timeout_seconds=30.0,
    )
