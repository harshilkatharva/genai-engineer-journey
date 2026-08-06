import os

from dotenv import load_dotenv

from llm_client.exceptions import ConfigError

load_dotenv()

try:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
    FREE_API = os.environ["FREE_API"]
except ConfigError as e:
    raise ConfigError(f"Missing required configuration: {e!s}")
