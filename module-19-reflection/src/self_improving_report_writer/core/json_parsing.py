"""JSON response sanitization and unwrapping helpers for LLM structured outputs."""

from __future__ import annotations

import json
import re
from typing import Any

# Matches opening markdown code fences such as ```json, ```JSON, or ```
_OPENING_FENCE_RE = re.compile(
    r"```(?:json|[a-zA-Z0-9_-]+(?=\s|$))?[ \t]*(?:\r?\n)?",
    re.IGNORECASE,
)
_TRAILING_FENCE_RE = re.compile(r"\s*```$")
_STANDALONE_CLOSING_FENCE_RE = re.compile(r"(?:\r?\n)[ \t]*```[ \t]*(?=\r?\n|$)")


def strip_markdown_json_fences(raw_text: str) -> str:
    """Strip outer Markdown code fences from a raw JSON text response.

    Supports ```` ```json ... ``` ````, ```` ```JSON ... ``` ````, untagged
    ```` ``` ... ``` ````, single-line fenced blocks, and fences surrounded by
    prose preambles/postambles while leaving valid raw JSON objects or arrays
    with embedded backticks untouched.
    """
    if not raw_text:
        return ""

    text = raw_text.strip()
    if not text or text[0] in ("{", "["):
        return text

    if text.startswith("```"):
        body = _OPENING_FENCE_RE.sub("", text, count=1).strip()
    else:
        open_match = _OPENING_FENCE_RE.search(text)
        if open_match is None:
            return text
        body = text[open_match.end() :].strip()

    if body.endswith("```"):
        return _TRAILING_FENCE_RE.sub("", body).strip()

    closing_matches = list(_STANDALONE_CLOSING_FENCE_RE.finditer(body))
    if closing_matches:
        return body[: closing_matches[-1].start()].strip()

    return body


def parse_json_with_fence_stripping(raw_text: str) -> Any:
    """Strip Markdown code fences and decode JSON into Python data."""
    cleaned = strip_markdown_json_fences(raw_text)
    if not cleaned:
        raise ValueError("Response text is empty after stripping Markdown code fences.")
    return json.loads(cleaned)


def unwrap_llm_response_payload(
    response: Any,
    *,
    allow_content_fallback: bool = False,
) -> Any:
    """Extract structured response or parse Markdown-fenced JSON from raw text.

    Args:
        response: Response object returned by LLMService.
        allow_content_fallback: Whether to inspect ``response.content`` when
            ``structured_response`` is ``None``.

    Returns:
        Model instance, parsed dict/JSON value, or ``None`` when no payload exists.
    """
    payload = getattr(response, "structured_response", None)
    if payload is None and allow_content_fallback and hasattr(response, "content"):
        payload = response.content

    if isinstance(payload, str):
        return parse_json_with_fence_stripping(payload) if payload.strip() else None

    if payload is not None:
        return payload

    raw_text = getattr(response, "text", None)
    if isinstance(raw_text, str) and raw_text.strip():
        return parse_json_with_fence_stripping(raw_text)

    return None


__all__ = [
    "parse_json_with_fence_stripping",
    "strip_markdown_json_fences",
    "unwrap_llm_response_payload",
]
