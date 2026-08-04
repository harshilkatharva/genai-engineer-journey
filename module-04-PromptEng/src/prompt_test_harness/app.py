from jinja2 import Template
from pathlib import Path
import json
from dotenv import load_dotenv
import os
from google import genai
from pydantic import BaseModel

load_dotenv()

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]


def get_answers_from_google_api(
    prompt: str, model: str | None = None, config: dict | None = None
) -> str:
    """
    Function to get answers from Google API using the provided prompt.

    Args:
        prompt (str): The input prompt for which the answer is to be generated.
        model (str): The model to use for generating the answer.
        config (dict | None): The configuration for generating the answer.

    Returns:
        str: The generated answer from the Google API.
    """
    response = google_client.models.generate_content(
        model=model if model else "gemini-3.5-flash-lite",
        contents=prompt,
        config=config if config else None,
    )

    if response:
        return response
    else:
        "Not answer from google"


def clean_response(text: str):
    clean = text.removeprefix("```json\n").removesuffix("\n```").strip()
    return clean


def schema_output(prompt: str, schema: BaseModel):
    try:
        response = get_answers_from_google_api(prompt)
        clean = clean_response(response.text)
        answer = json.loads(clean)
        answer = schema.model_validate(answer)
        return answer

    except Exception as e:
        prompt = (
            prompt
            + f"""
        i gave you this prompt and i specified output format but you did not work correctly
        Answer :- {response.text}

        Error :- {e}

        Try again and this time carefull in output fomate
        """
        )
        print("Retrying")
        # return schema_output(prompt, schema)


prompt_template = Template(
    Path("src/prompt_test_harness/prompts/recommandation_v1.md").read_text()
)


class RecommandSchema(BaseModel):
    genre: str
    movies: list[str]


schema = RecommandSchema.model_json_schema()


with open("src/prompt_test_harness/data/golden_dataset.json", "r") as f:
    data = json.loads(f.read())

google_client = genai.Client(api_key=GOOGLE_API_KEY)

test_result = []


for obj in data:
    query = obj["question"]

    prompt = prompt_template.render(
        recommandation_response_schema=schema, user_query=query
    )

    answer = schema_output(prompt, schema)

    test_result.append(1 if obj["expected_output"]["genre"] == answer.genre else 0)

print(test_result)
