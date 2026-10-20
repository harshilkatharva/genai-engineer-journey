from __future__ import annotations

import pytest

from genai_core.exceptions import ToolNotFoundError
from genai_core.tools import RiskLevel, ToolRegistry, tool


@pytest.fixture
def sample_tools():
    @tool(name="read_file", tags=["fs", "read"], risk_level=RiskLevel.READ_ONLY)
    def read_file(path: str) -> str:
        return f"content of {path}"

    @tool(
        name="write_file",
        tags=["fs", "write"],
        risk_level=RiskLevel.REVERSIBLE_WRITE,
    )
    def write_file(path: str, content: str) -> str:
        return f"wrote to {path}"

    @tool(
        name="delete_database",
        tags=["db", "danger"],
        risk_level=RiskLevel.IRREVERSIBLE,
    )
    def delete_database(db_name: str) -> str:
        return f"deleted {db_name}"

    return [read_file, write_file, delete_database]


def test_registry_registration_and_retrieval(sample_tools):
    registry = ToolRegistry(sample_tools)
    assert len(registry) == 3
    assert "read_file" in registry
    assert "write_file" in registry
    assert "delete_database" in registry

    t = registry.get("read_file")
    assert t is not None
    assert t.name == "read_file"

    t_fail = registry.get("nonexistent")
    assert t_fail is None

    with pytest.raises(ToolNotFoundError):
        registry.get_or_fail("nonexistent")


def test_duplicate_registration_guards(sample_tools):
    registry = ToolRegistry(sample_tools)

    @tool(name="read_file")
    def another_read():
        return "other"

    with pytest.raises(ValueError, match="already registered"):
        registry.register(another_read, override=False)

    # With override=True
    registry.register(another_read, override=True)
    assert registry.get("read_file") is another_read


def test_register_raw_callable():
    registry = ToolRegistry()

    def greet(name: str) -> str:
        return f"Hello, {name}"

    reg_tool = registry.register(greet)
    assert reg_tool.name == "greet"
    assert registry.has("greet")


def test_unregister_and_clear(sample_tools):
    registry = ToolRegistry(sample_tools)
    removed = registry.unregister("read_file")
    assert removed is not None
    assert removed.name == "read_file"
    assert not registry.has("read_file")
    assert len(registry) == 2

    registry.clear()
    assert len(registry) == 0


def test_get_specs_with_risk_and_tag_filtering(sample_tools):
    registry = ToolRegistry(sample_tools)

    # Filter max_risk = READ_ONLY
    read_only_specs = registry.get_specs(max_risk=RiskLevel.READ_ONLY)
    assert len(read_only_specs) == 1
    assert read_only_specs[0].name == "read_file"

    # Filter max_risk = REVERSIBLE_WRITE (includes READ_ONLY and REVERSIBLE_WRITE)
    reversible_specs = registry.get_specs(max_risk=RiskLevel.REVERSIBLE_WRITE)
    assert len(reversible_specs) == 2
    names = {s.name for s in reversible_specs}
    assert names == {"read_file", "write_file"}

    # Filter by tag
    db_specs = registry.get_specs(tags={"db"})
    assert len(db_specs) == 1
    assert db_specs[0].name == "delete_database"

    # Filter with exclude
    excluded_specs = registry.get_specs(exclude={"delete_database"})
    assert len(excluded_specs) == 2
    assert "delete_database" not in {s.name for s in excluded_specs}
