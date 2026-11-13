"""Token-to-USD pricing tables and deterministic cost estimation (FINDING-03)."""

from __future__ import annotations

from typing import Any

# Pricing per 1,000,000 tokens (USD) as (input_price_per_1m, output_price_per_1m)
MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    # Gemini models
    "gemini-3.5-flash-lite": (0.075, 0.30),
    "gemini-3.1-flash-lite": (0.075, 0.30),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-2.5-pro": (1.25, 5.00),
    # OpenAI models
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    # Default fallback pricing (conservative lightweight tier: $0.10 / $0.40 per 1M)
    "default": (0.10, 0.40),
}


def _resolve_pricing(model: str | None) -> tuple[float, float]:
    """Resolve input and output price per 1M tokens for a model."""
    if not model:
        return MODEL_PRICING_PER_1M["default"]

    normalized = model.strip().lower()
    # Strip any provider prefix (e.g., 'google/gemini-1.5-pro' -> 'gemini-1.5-pro')
    if "/" in normalized:
        normalized = normalized.split("/")[-1]

    if normalized in MODEL_PRICING_PER_1M:
        return MODEL_PRICING_PER_1M[normalized]

    # Partial / family match
    for known_model, pricing in MODEL_PRICING_PER_1M.items():
        if known_model in normalized or normalized in known_model:
            return pricing

    # Fallback to pro tier if 'pro' in model name, otherwise default
    if "pro" in normalized:
        return MODEL_PRICING_PER_1M["gemini-1.5-pro"]

    return MODEL_PRICING_PER_1M["default"]


def estimate_llm_cost(
    model: str | None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int | None = None,
) -> float:
    """Estimate USD cost based on model pricing and token counts.

    Args:
        model: Model name/identifier.
        input_tokens: Prompt/input token count.
        output_tokens: Completion/output token count.
        total_tokens: Total token count (used if input/output tokens are not specified).

    Returns:
        Estimated cost in USD rounded to 6 decimal places.
    """
    in_rate, out_rate = _resolve_pricing(model)

    in_tok = max(0, input_tokens)
    out_tok = max(0, output_tokens)

    # If only total_tokens is provided and individual counts are 0, estimate 70% input / 30% output
    if in_tok == 0 and out_tok == 0 and total_tokens and total_tokens > 0:
        in_tok = round(total_tokens * 0.70)
        out_tok = max(0, total_tokens - in_tok)

    input_cost = (in_tok / 1_000_000.0) * in_rate
    output_cost = (out_tok / 1_000_000.0) * out_rate
    return round(input_cost + output_cost, 6)


def extract_or_estimate_cost(
    response: Any,
    model: str | None = None,
    default_model: str | None = None,
) -> tuple[int, int, int, float]:
    """Extract token counts and cost in USD from an LLM response or estimate it.

    Args:
        response: LLMManagerResponse or object with usage/cost attributes.
        model: Explicit model passed to the request.
        default_model: Fallback model identifier.

    Returns:
        Tuple of (input_tokens, output_tokens, total_tokens, cost_usd).
    """
    usage = getattr(response, "usage", {}) or {}
    if isinstance(usage, dict):
        in_tok = int(usage.get("input_tokens", 0) or 0)
        out_tok = int(usage.get("output_tokens", 0) or 0)
        total_tok = int(usage.get("total_tokens", 0) or (in_tok + out_tok))
    else:
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        total_tok = int(getattr(usage, "total_tokens", 0) or (in_tok + out_tok))

    # Check for direct cost reported on response
    cost_val = getattr(response, "cost_usd", None)
    if cost_val is None and isinstance(getattr(response, "raw_response", None), dict):
        raw = response.raw_response
        cost_val = raw.get("cost") or raw.get("cost_usd")

    if cost_val is not None and float(cost_val) > 0.0:
        cost_usd = round(float(cost_val), 6)
    else:
        effective_model = getattr(response, "model", None) or model or default_model
        cost_usd = estimate_llm_cost(
            model=effective_model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=total_tok,
        )

    return in_tok, out_tok, total_tok, cost_usd
