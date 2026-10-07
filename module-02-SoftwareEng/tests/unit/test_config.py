import importlib

import pytest
from pytest import MonkeyPatch

import src.llm_client.config
from src.llm_client.exceptions import ConfigError


def test_missing_openai_api_key(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "dotenv.load_dotenv",
        lambda: None,
    )
    with pytest.raises(
        ConfigError,
        match="Missing required configuration: 'OPENAI_API_KEY'",
    ):
        importlib.reload(src.llm_client.config)
