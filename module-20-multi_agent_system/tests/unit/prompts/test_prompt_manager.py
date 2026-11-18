"""Unit tests for PromptTemplateManager."""

from __future__ import annotations

import pytest

from parallel_research_team.core.config import ResearchTeamConfig
from parallel_research_team.models import SourceReference, WorkerBrief
from parallel_research_team.prompts.manager import PromptTemplateManager


def test_prompt_manager_initialization(test_config: ResearchTeamConfig) -> None:
    mgr = PromptTemplateManager(config=test_config)
    assert mgr.default_version == "v1"
    versions = mgr.list_versions()
    assert "v1" in versions


def test_get_metadata(test_config: ResearchTeamConfig) -> None:
    mgr = PromptTemplateManager(config=test_config)
    meta = mgr.get_metadata("v1")
    assert meta.get("version") == "v1"
    assert "created_at" in meta


def test_render_decomposition_prompts(test_config: ResearchTeamConfig) -> None:
    mgr = PromptTemplateManager(config=test_config)
    system_prompt, user_prompt = mgr.render_decomposition_prompts(
        topic="Compare DuckDB and SQLite for local analytical queries",
        max_workers=3,
        effort_mode="parallel",
        user_preferences="Prefers columnar benchmarks",
    )

    assert "Lead Research Strategist" in system_prompt
    assert "DIRECT" in system_prompt
    assert "PARALLEL_DECOMPOSED" in system_prompt
    assert "DuckDB and SQLite" in user_prompt
    assert "Prefers columnar benchmarks" in user_prompt
    assert "3" in user_prompt


def test_render_worker_prompts(test_config: ResearchTeamConfig, sample_brief: WorkerBrief) -> None:
    mgr = PromptTemplateManager(config=test_config)
    system_prompt, user_prompt = mgr.render_worker_prompts(brief=sample_brief)

    assert "Research Worker" in system_prompt
    assert sample_brief.title in user_prompt
    assert sample_brief.objective in user_prompt
    assert "Focus strictly on" in user_prompt


def test_render_synthesis_prompts(
    test_config: ResearchTeamConfig, sample_sources: list[SourceReference]
) -> None:
    mgr = PromptTemplateManager(config=test_config)
    worker_findings = {
        "brief_01": "HNSW delivers lower latency.",
        "brief_02": "DiskANN scales to billions of vectors.",
    }
    system_prompt, user_prompt = mgr.render_synthesis_prompts(
        topic="Vector Indexing",
        worker_findings=worker_findings,
        sources=sample_sources,
    )

    assert "Lead Research Synthesizer" in system_prompt
    assert "Vector Indexing" in user_prompt
    assert "HNSW delivers lower latency" in user_prompt
    assert "src_vector_01" in user_prompt


def test_missing_template_raises(test_config: ResearchTeamConfig) -> None:
    mgr = PromptTemplateManager(config=test_config)
    with pytest.raises(FileNotFoundError, match="not found"):
        mgr.render_template("non_existent_folder/missing")
