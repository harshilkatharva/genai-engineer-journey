"""Prompt input sanitization helpers for untrusted XML data isolation."""

from __future__ import annotations

import html


def escape_xml_context(text: str) -> str:
    """Sanitize untrusted text embedded inside XML isolation delimiters.

    Escapes ``&``, ``<``, and ``>`` uniformly so embedded content cannot forge
    or prematurely close ``<untrusted_source_context>`` delimiters while
    keeping quotes readable inside body text.
    """
    if not text:
        return ""
    return html.escape(text, quote=False)


def escape_xml_attr(value: str) -> str:
    """Sanitize untrusted text placed inside XML tag attributes."""
    if not value:
        return ""
    return html.escape(value, quote=True)


def format_untrusted_source_block(content: str, **attributes: str) -> str:
    """Wrap untrusted content and metadata attributes in an escaped XML block."""
    clean_content = escape_xml_context(content)
    if attributes:
        attr_str = " ".join(
            f'{key}="{escape_xml_attr(str(val))}"' for key, val in attributes.items()
        )
        opening_tag = f"<untrusted_source_context {attr_str}>"
    else:
        opening_tag = "<untrusted_source_context>"
    return f"{opening_tag}\n{clean_content}\n</untrusted_source_context>"


__all__ = [
    "escape_xml_attr",
    "escape_xml_context",
    "format_untrusted_source_block",
]
