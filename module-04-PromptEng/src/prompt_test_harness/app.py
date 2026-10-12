import json
import os
import re
import time
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from jinja2 import Template
from pydantic import BaseModel

load_dotenv()

SchemaT = TypeVar("SchemaT", bound=BaseModel)

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

google_client = genai.Client(api_key=GOOGLE_API_KEY)


def get_answers_from_google_api(
    prompt: str,
    model: str | None = None,
    config: types.GenerateContentConfig | types.GenerateContentConfigDict | None = None,
) -> types.GenerateContentResponse:
    """
    Function to get answers from Google API using the provided prompt.

    Args:
        prompt (str): The input prompt for which the answer is to be generated.
        model (str): The model to use for generating the answer.
        config: The configuration for generating the answer.

    Returns:
        GenerateContentResponse: The response from the Google API.
    """
    return google_client.models.generate_content(
        model=model if model else "gemini-3.5-flash-lite",
        contents=prompt,
        config=config if config else None,
    )


def clean_response(text: str) -> str:
    clean = text.removeprefix("```json\n").removesuffix("\n```").strip()
    return clean


def schema_output(prompt: str, schema: type[SchemaT], retry_no: int = 0) -> SchemaT:
    try:
        response = get_answers_from_google_api(prompt)
    except errors.APIError as e:
        if e.code != 429:
            raise

        msg = str(e)
        match = re.search(r"retry in ([\d.]+)s", msg, re.IGNORECASE)
        wait_time = (int(float(match.group(1))) + 1 if match else 60) + 5

        print(f"Rate limit hit. Retrying in {wait_time} seconds...")
        time.sleep(wait_time)
        print(f"{wait_time} completed Retrying...")
        return schema_output(prompt, schema)

    response_text = response.text
    if response_text is None:
        raise ValueError("The model response did not contain text.")

    try:
        clean = clean_response(response_text)
        answer = json.loads(clean)
        return schema.model_validate(answer)

    except Exception as e:
        prompt = (
            prompt
            + f"""
            i gave you this prompt and i specified output format but you did not work correctly
            Answer :- {response_text}

            Error :- {e}

            Try again and this time carefull in output fomate
            """
        )
        if retry_no >= 1:
            raise

        print("Schema not matched Retrying...")
        return schema_output(prompt, schema, 1)


# Define Model response schema
class RecommandSchema(BaseModel):
    genre: str
    movies: list[str]


schema = RecommandSchema.model_json_schema()

# Load Golden dataset
with open("src/prompt_test_harness/data/golden_dataset.json", "r") as f:
    data = json.loads(f.read())


# Try with verison 1 prompt
prompt_template = Template(
    Path("src/prompt_test_harness/prompts/recommandation_v1.md").read_text()
)

test_result = []
answers = []

for obj in data:
    query = obj["question"]

    prompt = prompt_template.render(
        recommandation_response_schema=schema, user_query=query
    )
    answer = schema_output(prompt, RecommandSchema)
    print(f"Answer genrated of :- {query}")
    answers.append(answer)
    test_result.append(1 if obj["expected_output"]["genre"] == answer.genre else 0)

pass_rate = round((test_result.count(1) / len(test_result)) * 100, 2)
print("Test Result V1:- ", test_result)
print("Pass Rate V1:- ", pass_rate)

indices = [i for i, value in enumerate(test_result) if value == 0]

if indices:
    print("Wrong Results")

    for i in indices:
        print(f"Data :- {data[i]}")
        print(f"Answer :- {answers[i]}")


# Try with version 2 prompt
prompt_template_v2 = Template(
    Path("src/prompt_test_harness/prompts/recommandation_v2.md").read_text()
)

test_result_v2 = []
answers_v2 = []

for obj in data:
    query = obj["question"]

    prompt = prompt_template_v2.render(
        recommandation_response_schema=schema, user_query=query
    )
    answer = schema_output(prompt, RecommandSchema)
    print(f"V2 Answer genrated of :- {query}")
    answers_v2.append(answer)
    test_result_v2.append(1 if obj["expected_output"]["genre"] == answer.genre else 0)

print("Test Result :- ", test_result_v2)
pass_rate_v2 = round((test_result_v2.count(1) / len(test_result_v2)) * 100, 2)
print("Pass Rate :- ", pass_rate_v2)


indices_v2 = [i for i, value in enumerate(test_result_v2) if value == 0]

if indices_v2:
    print("Wrong Results")

    for i in indices_v2:
        print(f"Data :- {data[i]}")
        print(f"Answer :- {answers_v2[i]}")
