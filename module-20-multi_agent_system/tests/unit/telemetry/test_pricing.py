"""Unit tests for PricingCalculator."""

from __future__ import annotations

import pytest

from parallel_research_team.telemetry.pricing import (
    DEFAULT_FALLBACK_PRICE,
    PricingCalculator,
)


def test_pricing_calculator_exact_match() -> None:
    calc = PricingCalculator()
    price = calc.get_price("gemini-3.5-flash-lite")
    assert price.prompt_cost_per_million == 0.30
    assert price.completion_cost_per_million == 2.50


def test_pricing_calculator_normalization_and_prefix() -> None:
    calc = PricingCalculator()
    price = calc.get_price("google/gemini-3.1-flash-lite")
    assert price.prompt_cost_per_million == 0.25
    assert price.completion_cost_per_million == 1.50


def test_pricing_calculator_fallback() -> None:
    calc = PricingCalculator()
    price = calc.get_price("unknown-model-xyz")
    assert price == DEFAULT_FALLBACK_PRICE

    # Pro tier fallback
    pro_price = calc.get_price("custom-pro-model")
    assert pro_price.prompt_cost_per_million == 2.00


def test_calculate_cost() -> None:
    calc = PricingCalculator()
    # gemini-3.5-flash-lite: $0.30 / 1M prompt, $2.50 / 1M completion
    # 1,000,000 prompt tokens = $0.30
    # 500,000 completion tokens = $1.25
    # Total = $1.55
    cost = calc.calculate_cost(
        "gemini-3.5-flash-lite", input_tokens=1_000_000, output_tokens=500_000
    )
    assert cost == pytest.approx(1.55, rel=1e-5)

    # 10,000 prompt tokens = $0.003
    # 2,000 completion tokens = $0.005
    # Total = $0.008
    small_cost = calc.calculate_cost(
        "gemini-3.5-flash-lite", input_tokens=10_000, output_tokens=2_000
    )
    assert small_cost == pytest.approx(0.008, rel=1e-5)
