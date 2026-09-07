import time

from anthropic import Anthropic
from anthropic import APIError as ANTHROPIC_APIERROR
from google import genai
from google.genai.errors import APIError as GOOGLE_APIERROR
from openai import APIError as OPENAAI_APIERROR
from openai import OpenAI

from llm_client import config

# Fetching API Key from config
OPENAI_API_KEY = getattr(config, "OPENAI_API_KEY", None)
GOOGLE_API_KEY = getattr(config, "GOOGLE_API_KEY", None)
ANTHROPIC_API_KEY = getattr(config, "ANTHROPIC_API_KEY", None)
POSTGRESQL_CONNECTION = getattr(config, "POSTGRESQL_CONNECTION", False)
REDIS_CONNECTION = getattr(config, "REDIS_CONNECTION", False)


# Schema of result
results_schema = {
    "OpenAI": {
        "key_available": "Yes" if OPENAI_API_KEY else "No",
        "latency": None,
        "result": None,
    },
    "Google": {
        "key_available": "Yes" if GOOGLE_API_KEY else "No",
        "latency": None,
        "result": None,
    },
    "Anthropic": {
        "key_available": "Yes" if ANTHROPIC_API_KEY else "No",
        "latency": None,
        "result": None,
    },
}


class HealthCheck:
    def __init__(self):
        self.results = results_schema

    def check(self):
        if OPENAI_API_KEY:
            start = time.perf_counter()
            try:
                openai_client = OpenAI(api_key=OPENAI_API_KEY)
                openai_client.responses.create(model="gpt-4.1-mini", input="Hello World!")
                end = time.perf_counter()
                self.results["OpenAI"]["latency"] = end - start
                self.results["OpenAI"]["result"] = "success"

            except OPENAAI_APIERROR:
                end = time.perf_counter()
                self.results["OpenAI"]["latency"] = end - start
                self.results["OpenAI"]["result"] = "failed"

        # Google Check
        if GOOGLE_API_KEY:
            start = time.perf_counter()
            try:
                google_client = genai.Client(api_key=GOOGLE_API_KEY)
                google_client.interactions.create(
                    model="gemini-3.5-flash-lite", input="Hello World!"
                )
                end = time.perf_counter()
                self.results["Google"]["latency"] = end - start
                self.results["Google"]["result"] = "success"

            except GOOGLE_APIERROR:
                end = time.perf_counter()
                self.results["Google"]["latency"] = end - start
                self.results["Google"]["result"] = "failed"

        # Anthropic Check
        if ANTHROPIC_API_KEY:
            start = time.perf_counter()
            try:
                anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
                anthropic_client.messages.create(
                    model="claude-3.0",
                    messages=[{"role": "user", "content": "Hello World!"}],
                )
                end = time.perf_counter()
                self.results["Anthropic"]["latency"] = end - start
                self.results["Anthropic"]["result"] = "success"
            except ANTHROPIC_APIERROR:
                end = time.perf_counter()
                self.results["Anthropic"]["latency"] = end - start
                self.results["Anthropic"]["result"] = "failed"

        return self.results
