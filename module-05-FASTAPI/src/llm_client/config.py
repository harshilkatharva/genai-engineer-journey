import os

from dotenv import load_dotenv

from llm_client.exceptions import ConfigError

load_dotenv()

try:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
    X_API_KEY = os.environ["X_API_KEY"]
except KeyError as e:
    raise ConfigError(e.args[0])
