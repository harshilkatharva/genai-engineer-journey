"""Unit tests for pricing rate cards and cost calculation engine."""

from __future__ import annotations

from research_and_support_agent.telemetry.pricing import PricingCalculator


def test_pricing_calculator_gemini_35_flash_lite() -> None:
    """Verify calculation for Gemini 3.5 Flash Lite ($0.075 / 1M prompt, $0.30 / 1M completion)."""
    calc = PricingCalculator()
    cost = calc.calculate_cost("gemini-3.5-flash-lite", input_tokens=10_000, output_tokens=5_000)
    assert round(cost, 6) == 0.002250


def test_pricing_calculator_gemini_38_flash() -> None:
    """Verify calculation for Gemini 3.8 Flash rate card."""
    calc = PricingCalculator()
    cost = calc.calculate_cost("gemini-3.8-flash", input_tokens=10_000, output_tokens=5_000)
    assert round(cost, 6) == 0.003000


def test_pricing_calculator_gemini_15_flash() -> None:
    """Verify calculation for Gemini 1.5 Flash."""
    calc = PricingCalculator()
    cost = calc.calculate_cost("gemini-1.5-flash", input_tokens=10_000, output_tokens=5_000)
    assert round(cost, 6) == 0.002250


def test_pricing_calculator_gpt4o() -> None:
    """Verify calculation for GPT-4o ($2.50 / 1M prompt, $10.00 / 1M completion)."""
    calc = PricingCalculator()
    cost = calc.calculate_cost("gpt-4o", input_tokens=1_000, output_tokens=1_000)
    assert round(cost, 6) == 0.012500


def test_pricing_calculator_unknown_model_fallback() -> None:
    """Verify unknown model falls back safely to default rate without throwing."""
    calc = PricingCalculator()
    cost = calc.calculate_cost("some-custom-local-model", input_tokens=10_000, output_tokens=10_000)
    assert cost > 0.0


def test_pricing_zero_tokens() -> None:
    """Verify zero token usage yields $0.00 cost."""
    calc = PricingCalculator()
    assert calc.calculate_cost("gemini-3.5-flash-lite", 0, 0) == 0.0
