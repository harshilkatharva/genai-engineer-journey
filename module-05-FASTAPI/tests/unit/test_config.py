import importlib

import pytest
from pytest import MonkeyPatch

import llm_client.config
from llm_client.exceptions import ConfigError


def test_missing_openai_api_key(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    monkeypatch.setattr("dotenv.main.find_dotenv", lambda *args, **kwargs: "")

    with pytest.raises(
        ConfigError,
        match="Missing required configuration: 'OPENAI_API_KEY'",
    ):
        importlib.reload(llm_client.config)
