from ..logger.llm_calls_logger import llm_call_logger
from ..providers.google_provider import GoogleProvider
from ..models.llm_response import LLMResponseModel
from google.genai.errors import ClientError, ServerError
from pydantic import ValidationError

import json
from jinja2 import Template
from pathlib import Path
import time
import re


class Prompt_v2:
    def __init__(self):
        with open("src/assignment/data/persons_details.json", "r") as f:
            self.data = json.loads(f.read())
            self.google_provider = GoogleProvider()
            self.model = LLMResponseModel
            self.model_schema = LLMResponseModel.model_json_schema()

    def _clean_response(self, text: str):
        clean = text.removeprefix("```json\n").removesuffix("\n```").strip()
        clean = json.loads(clean)
        return clean

    def _get_output(self, prompt: str, retry_no: int = 0):
        retry = True
        while True:
            try:
                response = self.google_provider.get_answer(prompt)
                clean = self._clean_response(response.text)
                try:
                    answer = self.model.model_validate(clean)
                    return answer
                except ValidationError:
                    return clean
            except ClientError as e:
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
            except ServerError:
                print("Model busy. Retrying in 10 seconds...")
                time.sleep(10)
                continue

    def process(self):
        prompt_template = Template(
            Path("src/assignment/prompts/details_extration_v2.md").read_text()
        )
        answers = []
        for obj in self.data:
            prompt = f"{prompt_template.render()}  {obj['input']}"
            response = self._get_output(prompt)
            print("answer llm")
            llm_call_logger.info(
                f"\nID :- {obj['id']}, Question :- {obj['input']}, Response :- {response}\n"
            )
            answers.append(
                {"id": obj["id"], "Question": obj["input"], "Response": response}
            )

        with open("src/assignment/data/prompt_v2.json", "a") as f:
            json.dump(answers, f, indent=4, ensure_ascii=False)

    def check_result(self):
        with open("src/assignment/data/prompt_v2.json", "r") as f:
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
