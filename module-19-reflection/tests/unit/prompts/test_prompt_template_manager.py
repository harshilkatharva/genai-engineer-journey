"""Unit tests for PromptTemplateManager and Jinja2 environment."""

from __future__ import annotations

from pathlib import Path

import pytest

from self_improving_report_writer.prompts.manager import (
    PromptTemplateManager,
    escape_xml_context,
    get_prompt_manager,
)


def test_prompt_manager_initialization() -> None:
    """Test default initialization and singleton retrieval."""
    mgr = get_prompt_manager()
    assert isinstance(mgr, PromptTemplateManager)
    assert mgr.default_version == "v1"
    assert mgr.template_dir.exists()
    assert (mgr.template_dir / "v1").exists()


def test_prompt_manager_list_templates() -> None:
    """Test listing of all available templates in v1."""
    mgr = get_prompt_manager()
    templates = mgr.list_templates("v1")
    assert len(templates) >= 10

    expected_templates = [
        "generation/system.jinja2",
        "generation/user.jinja2",
        "judging/system.jinja2",
        "judging/user.jinja2",
        "revision/system.jinja2",
        "revision/user.jinja2",
        "verification/batch_claim_entailment_user.jinja2",
        "verification/claim_entailment_system.jinja2",
        "verification/claim_entailment_user.jinja2",
        "verification/prose_consistency_system.jinja2",
        "verification/prose_consistency_user.jinja2",
    ]
    for tpl in expected_templates:
        assert tpl in templates, f"Missing expected template: {tpl}"


def test_prompt_manager_metadata() -> None:
    """Test metadata reading for v1."""
    mgr = get_prompt_manager()
    meta = mgr.get_version_metadata("v1")
    assert meta["version"] == "v1"
    assert "description" in meta


def test_prompt_manager_custom_dir(tmp_path: Path) -> None:
    """Test PromptTemplateManager with custom directory."""
    custom_v1 = tmp_path / "v1" / "custom"
    custom_v1.mkdir(parents=True)
    tpl_file = custom_v1 / "hello.jinja2"
    tpl_file.write_text("Hello {{ name }}!", encoding="utf-8")

    mgr = PromptTemplateManager(template_dir=tmp_path, default_version="v1")
    rendered = mgr.render_template("custom/hello", name="World")
    assert rendered == "Hello World!"


def test_prompt_manager_missing_template() -> None:
    """Test FileNotFoundError is raised for non-existent template."""
    mgr = get_prompt_manager()
    with pytest.raises(FileNotFoundError, match="not found"):
        mgr.render_template("non_existent/template")


def test_prompt_manager_escape_xml_filter() -> None:
    """Test custom escape_xml filter in Jinja2 environment."""
    mgr = get_prompt_manager()
    template = mgr.jinja_env.from_string("Safe: {{ raw | escape_xml }}")
    rendered = template.render(raw="A & B < C > D")
    assert rendered == "Safe: A &amp; B &lt; C &gt; D"


def test_escape_xml_context() -> None:
    """Test escape_xml_context helper function."""
    assert escape_xml_context("a & b < c > d") == "a &amp; b &lt; c &gt; d"
    assert escape_xml_context("clean text") == "clean text"
