from jinja2 import Template
from pathlib import Path
import json
from dotenv import load_dotenv
import os
from google import genai
from pydantic import BaseModel
import time
import re


load_dotenv()

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

google_client = genai.Client(api_key=GOOGLE_API_KEY)


PROMPT_DIR = Path("src/prompt_test_harness/prompts")
DATASET_PATH = Path("src/prompt_test_harness/data/golden_dataset.json")


# ============================================================
# API
# ============================================================


def get_answers_from_google_api(
    prompt: str,
    model: str | None = None,
    config: dict | None = None,
):
    response = google_client.models.generate_content(
        model=model or "gemini-3.5-flash-lite",
        contents=prompt,
        config=config,
    )

    return response


# ============================================================
# Response cleaning
# ============================================================


def clean_response(text: str) -> str:
    return text.removeprefix("```json").removesuffix("```").strip()


# ============================================================
# Structured output
# ============================================================


def schema_output(
    prompt: str,
    schema: type[BaseModel],
    retry_no: int = 0,
):
    try:
        response = get_answers_from_google_api(prompt)

        clean = clean_response(response.text)

        answer = json.loads(clean)

        return schema.model_validate(answer)

    except Exception as e:
        # Handle rate limiting
        if getattr(e, "code", None) == 429:
            msg = str(e)

            match = re.search(
                r"retry in ([\d.]+)s",
                msg,
                re.IGNORECASE,
            )

            wait_time = (int(float(match.group(1))) + 1 if match else 60) + 5

            print(f"Rate limit hit. Retrying in {wait_time} seconds...")

            time.sleep(wait_time)

            return schema_output(
                prompt,
                schema,
                retry_no,
            )

        # Retry once for invalid schema/output
        if retry_no < 1:
            retry_prompt = f"""
I gave you this prompt:

{prompt}

But your response did not match the required output format.

Your answer was:

{response.text if "response" in locals() else "No response"}

The error was:

{e}

Try again and carefully follow the required output format.
"""

            print("Schema not matched. Retrying...")

            return schema_output(
                retry_prompt,
                schema,
                retry_no + 1,
            )

        raise e


# ============================================================
# Schema
# ============================================================


class RecommandSchema(BaseModel):
    genre: str
    movies: list[str]


# ============================================================
# Direct recommendation function
# ============================================================


def get_recommendation(
    user_query: str,
    version: int = 1,
) -> RecommandSchema:
    prompt_path = PROMPT_DIR / f"recommandation_v{version}.md"

    prompt_template = Template(prompt_path.read_text())

    schema = RecommandSchema.model_json_schema()

    prompt = prompt_template.render(
        recommandation_response_schema=schema,
        user_query=user_query,
    )

    return schema_output(
        prompt,
        RecommandSchema,
    )


# ============================================================
# Load golden dataset
# ============================================================


def load_golden_dataset():
    with open(DATASET_PATH, "r") as f:
        return json.load(f)


# ============================================================
# Evaluate one prompt version
# ============================================================


def evaluate_prompt(version: int = 1):
    data = load_golden_dataset()

    results = []
    answers = []

    for obj in data:
        query = obj["question"]

        try:
            answer = get_recommendation(
                user_query=query,
                version=version,
            )

            is_correct = answer.genre == obj["expected_output"]["genre"]

            results.append(1 if is_correct else 0)
            answers.append(answer)

        except Exception as e:
            results.append(0)
            answers.append({"error": str(e)})

    pass_rate = round(
        (sum(results) / len(results)) * 100,
        2,
    )

    return {
        "version": version,
        "results": results,
        "answers": answers,
        "pass_rate": pass_rate,
    }


# ============================================================
# Evaluate all prompt versions
# ============================================================


def run_evaluation():
    v1 = evaluate_prompt(version=1)
    v2 = evaluate_prompt(version=2)

    return {
        "v1": v1,
        "v2": v2,
    }
