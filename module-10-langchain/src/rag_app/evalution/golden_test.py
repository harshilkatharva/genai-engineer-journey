import json
import time
import requests
from rag_app.core.settings import get_settings


API_URL = "http://127.0.0.1:8000/rag/chat_answer"

TENANT_ID = "06f197fb-3b03-469f-b3ba-461dae52cf7a"

settings = get_settings()

DATASET_PATH = settings.evalution_dataset


def call_queries():
    # Load dataset
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    questions = dataset["questions"]

    # Call each query one by one
    for index, item in enumerate(questions, start=1):
        query = item["query"]

        payload = {"tenant_id": TENANT_ID, "query": query}

        print("=" * 80)
        print(f"Running question {index}/{len(questions)}")
        print(f"Query: {query}")

        try:
            response = requests.post(API_URL, json=payload, timeout=120)

            print(f"Status Code: {response.status_code}")

            # try:
            #     print("Response:")
            #     print(json.dumps(response.json(), indent=2, ensure_ascii=False))
            # except ValueError:
            #     print("Response:")
            #     print(response.text)

        except requests.exceptions.RequestException as error:
            print(f"Error: {error}")

        print()

        # Optional delay between requests
        time.sleep(10)

    print("All queries completed.")
