from ..logger.llm_calls_logger import llm_call_logger
from ..providers.google_provider import GoogleProvider

import json
from jinja2 import Template
from pathlib import Path
import time
import re


class Prompt_v1:
    def __init__(self):
        with open("src/assignment/data/persons_details.json", "r") as f:
            self.data = json.loads(f.read())
            self.google_provider = GoogleProvider()

    def process(self):
        prompt_template = Template(
            Path("src/assignment/prompts/details_extration_v1.md").read_text()
        )
        answers = []
        for obj in self.data:
            prompt = f"{prompt_template.render()}  {obj['input']}"
            response = self._get_output(prompt)
            llm_call_logger.info(
                f"\nID :- {obj['id']}, Question :- {obj['input']}, Response :- {response}\n"
            )
            answers.append(
                {"id": obj["id"], "Question": obj["input"], "Response": response}
            )

        with open("src/assignment/data/prompt_v1.json", "a") as f:
            json.dump(answers, f, indent=4, ensure_ascii=False)

    def _get_output(self, prompt: str):
        retry = True
        while True:
            try:
                response = self.google_provider.get_answer(prompt)
                return response.text
            except Exception as e:
                if e.code == 429 and retry:
                    retry = False
                    msg = str(e)

                    match = re.search(r"retry in ([\d.]+)s", msg, re.IGNORECASE)
                    wait_time = (int(float(match.group(1))) + 1 if match else 60) + 5

                    print(f"Rate limit hit. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    llm_call_logger.warning(
                        f"Rate limit exceeded. Retrying in {wait_time} seconds..."
                    )
                    print(f"{wait_time} completed Retrying...")
                    continue
                else:
                    llm_call_logger.error(str(e))
                    raise e

    def check_result(self):
        with open("src/assignment/data/prompt_v1.json", "r") as f:
            results = json.loads(f.read())
        golden_lookup = {item["id"]: item for item in self.data}
        response_result = []
        compare = []
        for result in results:
            expected = golden_lookup.get(result["id"])
            if expected["expected_output"] == result["Response"]:
                response_result.append(1)
            else:
                response_result.append(0)
                compare.append(
                    [
                        {
                            "Expected": expected["expected_output"],
                            "Assistant": result["Response"],
                        }
                    ]
                )

        pass_percentage = round(
            (response_result.count(1) / len(response_result)) * 100, 2
        )
        return pass_percentage, compare
