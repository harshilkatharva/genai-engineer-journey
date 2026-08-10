from ai_app.core.cost_tracker import CostTracker


def test_cost_estimate():
    tracker = CostTracker()

    estimated_cost = tracker.get_cost(
        input_token=1000000, output_token=1000000, model="gemini-3.5-flash-lite"
    )

    assert estimated_cost == 2.8
