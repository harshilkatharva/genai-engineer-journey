"""Model pricing rate cards and cost calculation engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    """Per-million token pricing rate in USD."""

    prompt_cost_per_million: float
    completion_cost_per_million: float


# Default pricing catalog (USD per 1M tokens)
PRICING_TABLE: dict[str, ModelPrice] = {
    # Google Gemini Models
    "gemini-1.5-flash": ModelPrice(0.075, 0.30),
    "gemini-1.5-flash-8b": ModelPrice(0.0375, 0.15),
    "gemini-1.5-pro": ModelPrice(1.25, 5.00),
    "gemini-2.0-flash": ModelPrice(0.10, 0.40),
    # OpenAI Models
    "gpt-4o-mini": ModelPrice(0.15, 0.60),
    "gpt-4o": ModelPrice(2.50, 10.00),
    # Anthropic Models
    "claude-3-5-sonnet": ModelPrice(3.00, 15.00),
    "claude-3-5-haiku": ModelPrice(0.80, 4.00),
}

DEFAULT_PRICE = ModelPrice(0.075, 0.30)  # Default fallback matching Gemini 1.5 Flash


class PricingCalculator:
    """Calculates precise monetary cost in USD from token usage."""

    def __init__(self, table: dict[str, ModelPrice] | None = None) -> None:
        self.table = table or PRICING_TABLE

    def get_price(self, model_name: str | None) -> ModelPrice:
        """Find matching price card for model name (case-insensitive substring match)."""
        if not model_name:
            return DEFAULT_PRICE

        normalized = model_name.strip().lower()
        # Direct lookup
        if normalized in self.table:
            return self.table[normalized]

        # Substring prefix/suffix matching
        for key, price in self.table.items():
            if key in normalized or normalized in key:
                return price

        return DEFAULT_PRICE

    def calculate_cost(
        self,
        model_name: str | None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> float:
        """Calculate total USD cost for a given token usage breakdown."""
        price = self.get_price(model_name)
        input_cost = (max(0, input_tokens) / 1_000_000.0) * price.prompt_cost_per_million
        output_cost = (max(0, output_tokens) / 1_000_000.0) * price.completion_cost_per_million
        return round(input_cost + output_cost, 8)
