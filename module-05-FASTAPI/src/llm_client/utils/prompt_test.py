import asyncio
import json
import re
from pathlib import Path

import aiofiles
from google.genai.errors import ClientError, ServerError
from jinja2 import Template
from pydantic import ValidationError

from llm_client.models import DetailsExtractionModel
from llm_client.services.llm_service import LLMClient


class PromptTest:
    def __init__(self, client: LLMClient):
        with open("src/llm_client/data/golden_dataset.json", "r") as f:
            self.data = json.loads(f.read())
        self.client = client
        self.model = DetailsExtractionModel
        self.model_schema = DetailsExtractionModel.model_json_schema()

    def _clean_response(self, text: str):
        clean = text.removeprefix("```json\n").removesuffix("\n```").strip()
        clean = json.loads(clean)
        return clean

    async def _get_output(self, provider: str, prompt: str, retry_no: int = 0):
        retry = True
        while True:
            try:
                response = await self.client.complete(provider, prompt)
                clean = self._clean_response(response.text)
                try:
                    answer = self.model.model_validate(clean)
                    return answer.model_dump()
                except ValidationError:
                    return clean
            except ClientError as e:
                if e.code == 429 and retry:
                    retry = False
                    msg = str(e)

                    match = re.search(r"retry in ([\d.]+)s", msg, re.IGNORECASE)
                    wait_time = (int(float(match.group(1))) + 1 if match else 60) + 5

                    print(f"Rate limit hit. Retrying in {wait_time} seconds...")
                    asyncio.sleep(wait_time)
                    print(f"{wait_time} completed Retrying...")
                    continue
            except ServerError:
                print("Model busy. Retrying in 10 seconds...")
                asyncio.sleep(10)
                continue
            except Exception:
                raise

    async def process(self, provider: str):
        prompt_template = Template(
            Path("src/llm_client/prompts/details_extration_v3.md").read_text()
        )
        answers = []
        for obj in self.data:
            prompt = f"{prompt_template.render(ouput_schema=self.model_schema, text=obj['input'])}"
            response = await self._get_output(provider, prompt)
            print("answer llm")
            answers.append({"id": obj["id"], "Question": obj["input"], "Response": response})

        async with aiofiles.open("src/llm_client/data/prompt_result.json", "w") as f:
            await f.write(json.dump(answers, f, indent=4, ensure_ascii=False))

    def check_result(self):
        with open("src/llm_client/data/prompt_result.json", "r") as f:
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

        pass_percentage = round((response_result.count(1) / len(response_result)) * 100, 2)
        return {"Pass Percentage": pass_percentage, "Comparison Result ": compare}
