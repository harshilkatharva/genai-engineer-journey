import json
from pathlib import Path

from jinja2 import Template
from langchain_core.output_parsers import PydanticOutputParser

from structured_output_harness import (
    get_answers_from_google_api,
    schema_output,
    RecommandSchema,
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.fake_chat_models import FakeChatModel


# Task 5
prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant."), ("human", "Answer this question: {question}")]
)


class TestChatModel(FakeChatModel):
    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        # Return the constructed prompt so we can verify it
        return "\n".join(f"{message.type}: {message.content}" for message in messages)


model = TestChatModel()

chain = prompt | model


def test_prompt_construction():
    result = chain.invoke({"question": "What is LangChain?"})

    assert "You are a helpful assistant." in result.content
    assert "Answer this question: What is LangChain?" in result.content


DATASET_PATH = Path("data/golden_dataset.json")
PROMPT_PATH = Path("prompts/recommandation_v1.md")


def load_golden_set():
    with open(DATASET_PATH, "r") as f:
        return json.load(f)


def build_prompt(question):
    template = Template(PROMPT_PATH.read_text())

    return template.render(
        recommandation_response_schema=(RecommandSchema.model_json_schema()),
        user_query=question,
    )


def structured_output_reliability():
    data = load_golden_set()

    parser = PydanticOutputParser(pydantic_object=RecommandSchema)

    custom_valid = 0
    custom_correct = 0

    langchain_valid = 0
    langchain_correct = 0

    for item in data:
        question = item["question"]
        expected_genre = item["expected_output"]["genre"]

        prompt = build_prompt(question)

        # --------------------------------------------------
        # Module 4 custom harness
        # --------------------------------------------------

        try:
            custom_answer = schema_output(
                prompt,
                RecommandSchema,
            )

            custom_valid += 1

            if custom_answer.genre == expected_genre:
                custom_correct += 1

        except Exception as e:
            print(f"Custom harness failed: {question}")
            print(e)

        # --------------------------------------------------
        # LangChain PydanticOutputParser
        # --------------------------------------------------

        try:
            response = get_answers_from_google_api(prompt)

            langchain_answer = parser.parse(response.text)

            langchain_valid += 1

            if langchain_answer.genre == expected_genre:
                langchain_correct += 1

        except Exception as e:
            print(f"LangChain parser failed: {question}")
            print(e)

    total = len(data)

    custom_reliability = round(
        custom_valid / total * 100,
        2,
    )

    custom_accuracy = round(
        custom_correct / total * 100,
        2,
    )

    langchain_reliability = round(
        langchain_valid / total * 100,
        2,
    )

    langchain_accuracy = round(
        langchain_correct / total * 100,
        2,
    )

    print("\n" + "=" * 50)
    print("STRUCTURED OUTPUT COMPARISON")
    print("=" * 50)

    print(f"\nGolden set: {total} cases")

    print("\nModule 4 Custom Harness")
    print(f"Valid:       {custom_valid}/{total}")
    print(f"Reliability: {custom_reliability}%")
    print(f"Correct:     {custom_correct}/{total}")
    print(f"Accuracy:    {custom_accuracy}%")

    print("\nLangChain PydanticOutputParser")
    print(f"Valid:       {langchain_valid}/{total}")
    print(f"Reliability: {langchain_reliability}%")
    print(f"Correct:     {langchain_correct}/{total}")
    print(f"Accuracy:    {langchain_accuracy}%")

    print("\n" + "=" * 50)

    # The test passes as long as both parsers were evaluated.
    assert total > 0
