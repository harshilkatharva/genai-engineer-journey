from internal_tools_mcp import integrations


def test_repository_root_points_to_handbook_parent():
    assert integrations.REPOSITORY_ROOT.name == "genai-engineer-journey"


def test_add_source_inserts_path_once(tmp_path, monkeypatch):
    source = tmp_path / "src"
    source_string = str(source)
    monkeypatch.setattr(integrations.sys, "path", ["existing"])

    integrations._add_source(source)
    integrations._add_source(source)

    assert integrations.sys.path == [source_string, "existing"]
