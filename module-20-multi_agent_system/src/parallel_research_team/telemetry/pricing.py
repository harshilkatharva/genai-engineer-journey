"""Model pricing rate cards and USD cost calculation engine for Module 20."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    """Per-million token pricing rate in USD."""

    prompt_cost_per_million: float
    completion_cost_per_million: float


# Default pricing catalog (USD per 1,000,000 tokens)
PRICING_TABLE: dict[str, ModelPrice] = {
    # Google Gemini Models (USD per 1M tokens)
    "gemini-4-carbon": ModelPrice(
        2.00, 10.00
    ),  # Intro rate through Nov 30, 2026 ($4.00 / $20.00 starting Dec 1, 2026)
    "gemini-4-argon": ModelPrice(
        2.00, 10.00
    ),  # Intro rate through Nov 30, 2026 ($4.00 / $20.00 starting Dec 1, 2026)
    "gemini-4-hydrogen": ModelPrice(
        0.75, 3.75
    ),  # Vertex AI rate ($1.50 / $7.50 listed on some AI Studio endpoints)
    "gemini-3.8-flash": ModelPrice(
        0.75, 3.75
    ),  # Intro rate through Dec 31, 2026 ($1.50 / $7.50 starting Jan 1, 2027)
    "gemini-3.6-flash": ModelPrice(
        0.75, 3.75
    ),  # Intro rate through Dec 31, 2026 ($1.50 / $7.50 starting Jan 1, 2027)
    "gemini-3.5-flash-lite": ModelPrice(0.30, 2.50),
    "gemini-3.1-flash-lite": ModelPrice(0.25, 1.50),
    "gemini-3.1-pro-preview": ModelPrice(2.00, 12.00),  # <=200K context ($4.00 / $18.00 above 200K)
    # OpenAI Models (USD per 1M tokens)
    "gpt-6.2-astra": ModelPrice(8.00, 40.00),
    "gpt-6.2-sol": ModelPrice(2.00, 10.00),
    "gpt-6.1-terra": ModelPrice(1.00, 5.00),
    "gpt-6-luna": ModelPrice(0.10, 0.50),
    # Anthropic Models (USD per 1M tokens)
    "claude-fable-5-2": ModelPrice(8.00, 40.00),
    "claude-opus-5-5": ModelPrice(4.00, 20.00),
    "claude-sonnet-5-5": ModelPrice(2.00, 10.00),
    "claude-haiku-5-5": ModelPrice(0.10, 0.50),  # <=100K context ($0.50 / $2.50 above 100K)
}

DEFAULT_FALLBACK_PRICE = ModelPrice(0.10, 0.40)  # Conservative fallback ($0.10 / $0.40 per 1M)


class PricingCalculator:
    """Calculates monetary cost in USD from token usage."""

    def __init__(self, table: dict[str, ModelPrice] | None = None) -> None:
        self.table = table or PRICING_TABLE

    def get_price(self, model_name: str | None) -> ModelPrice:
        """Find matching price card for a model name using case-insensitive lookup."""
        if not model_name:
            return DEFAULT_FALLBACK_PRICE

        normalized = model_name.strip().lower()
        if "/" in normalized:
            normalized = normalized.split("/")[-1]

        # Exact match
        if normalized in self.table:
            return self.table[normalized]

        # Substring / family match
        for known_model, price in self.table.items():
            if known_model in normalized or normalized in known_model:
                return price

        # Pro model fallback
        if "pro" in normalized:
            return self.table.get("gemini-3.1-pro-preview", ModelPrice(2.00, 12.00))

        return DEFAULT_FALLBACK_PRICE

    def calculate_cost(
        self,
        model_name: str | None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> float:
        """Calculate USD expenditure based on token consumption."""
        price = self.get_price(model_name)
        input_cost = (max(0, input_tokens) / 1_000_000.0) * price.prompt_cost_per_million
        output_cost = (max(0, output_tokens) / 1_000_000.0) * price.completion_cost_per_million
        return round(input_cost + output_cost, 6)
