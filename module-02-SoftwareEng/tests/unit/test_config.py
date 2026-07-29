import pytest
import importlib
from llm_client.exceptions import ConfigError
import llm_client.config
from pytest import MonkeyPatch


def test_missing_openai_api_key(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(
        ConfigError,
        match="Missing required configuration: 'OPENAI_API_KEY'",
    ):
        importlib.reload(llm_client.config)
