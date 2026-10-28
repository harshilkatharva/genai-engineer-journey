"""Token counting and truncation utilities powered by tiktoken."""

from __future__ import annotations

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=8)
def get_tokenizer(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """Retrieve and cache a tiktoken Encoding instance."""
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in a string using the specified tiktoken encoding."""
    if not text:
        return 0
    enc = get_tokenizer(encoding_name)
    return len(enc.encode(text, disallowed_special=()))


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    encoding_name: str = "cl100k_base",
    suffix: str = "",
) -> tuple[str, bool]:
    """Truncate text to at most max_tokens using tiktoken.

    Args:
        text: Input string.
        max_tokens: Maximum allowed token count.
        encoding_name: Tiktoken encoding name (default: cl100k_base).
        suffix: Suffix to append if truncation occurs.

    Returns:
        Tuple of (result_text, was_truncated).
    """
    if not text or max_tokens <= 0:
        return "", False

    enc = get_tokenizer(encoding_name)
    tokens = enc.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text, False

    truncated_tokens = tokens[:max_tokens]
    truncated_text = enc.decode(truncated_tokens) + suffix
    return truncated_text, True
