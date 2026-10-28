"""Unit tests for tiktoken-powered token counting and truncation utilities."""

from __future__ import annotations

import tiktoken

from plan_and_execute_research_assistant.core.tokens import (
    count_tokens,
    get_tokenizer,
    truncate_to_tokens,
)


def test_get_tokenizer_caches() -> None:
    """Verify tokenizer instance is cached."""
    tok1 = get_tokenizer("cl100k_base")
    tok2 = get_tokenizer("cl100k_base")
    assert tok1 is tok2
    assert isinstance(tok1, tiktoken.Encoding)


def test_count_tokens() -> None:
    """Verify count_tokens returns accurate token count."""
    assert count_tokens("") == 0
    assert count_tokens("Hello world") == 2
    # Verify special tokens do not crash
    assert count_tokens("Hello <|endoftext|> world") > 2


def test_truncate_to_tokens_not_truncated() -> None:
    """Verify text under max_tokens is returned unchanged."""
    text = "Short sentence for testing."
    result, truncated = truncate_to_tokens(text, max_tokens=20)
    assert result == text
    assert truncated is False


def test_truncate_to_tokens_truncated() -> None:
    """Verify text exceeding max_tokens is truncated accurately."""
    words = "word " * 100
    suffix = "\n... [Truncated]"
    result, truncated = truncate_to_tokens(words, max_tokens=10, suffix=suffix)
    assert truncated is True
    assert result.endswith(suffix)

    # Token count of truncated content without suffix should be <= 10
    body = result[: -len(suffix)]
    assert count_tokens(body) <= 10
