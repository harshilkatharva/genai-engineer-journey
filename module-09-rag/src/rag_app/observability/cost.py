from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Cost:
    """Represents the cost of one operation."""

    amount_usd: float = 0.0

    def add(self, amount_usd: float) -> None:
        self.amount_usd += amount_usd


def calculate_token_cost(
    input_tokens: int,
    output_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float,
) -> float:
    """
    Calculate cost using USD per million tokens.
    """

    input_cost = (input_tokens / 1_000_000) * input_price_per_million

    output_cost = (output_tokens / 1_000_000) * output_price_per_million

    return input_cost + output_cost
