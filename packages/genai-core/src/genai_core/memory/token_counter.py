from __future__ import annotations

import math
from typing import Any

from ..models.memory_model import ConversationMessage


class HeuristicTokenCounter:
    """Character-based heuristic token counter (~4 characters per token).

    Does not require any external tokenizer libraries.
    """

    def __init__(self, chars_per_token: float = 4.0) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        """Count estimated tokens from string length."""
        if not text:
            return 0
        return max(1, math.ceil(len(text) / self.chars_per_token))

    def count_message(self, message: ConversationMessage) -> int:
        """Count tokens for a message including role overhead."""
        # Standard ~4 token message framing overhead
        formatted = f"{message.role}: {message.content}\n"
        return self.count(formatted) + 3


class TiktokenCounter:
    """Accurate token counter using OpenAI's tiktoken library if installed."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        try:
            import tiktoken  # type: ignore[import-not-found]

            self._encoding: Any = tiktoken.get_encoding(encoding_name)
        except ImportError as err:
            raise RuntimeError(
                "tiktoken is not installed. Install tiktoken or use HeuristicTokenCounter."
            ) from err

    def count(self, text: str) -> int:
        """Count tokens using tiktoken encoding."""
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def count_message(self, message: ConversationMessage) -> int:
        """Count tokens for a message including role overhead."""
        formatted = f"{message.role}: {message.content}\n"
        return self.count(formatted) + 3


def get_default_token_counter() -> HeuristicTokenCounter | TiktokenCounter:
    """Return TiktokenCounter if tiktoken is available, otherwise HeuristicTokenCounter."""
    try:
        import tiktoken  # type: ignore[import-not-found]  # noqa: F401

        return TiktokenCounter()
    except Exception:  # noqa: BLE001
        return HeuristicTokenCounter()
