from ai_app.utils.llm_model_cost_chart import cost_chart


class CostTracker:
    def __init__(self, cost_chart: dict = cost_chart):
        self.cost_chart = cost_chart

    def get_cost(self, input_token: int, output_token: int, model: str) -> float:
        if model not in self.cost_chart:
            raise KeyError("Model not found in cost chart.")
        input_price_per_token = self.cost_chart[model]["input"]
        output_price_per_token = self.cost_chart[model]["output"]

        input_price = input_token * input_price_per_token
        output_price = output_token * output_price_per_token

        return input_price + output_price
