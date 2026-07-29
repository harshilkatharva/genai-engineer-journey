import os
from llm_client.exceptions import ConfigError
from dotenv import load_dotenv

load_dotenv()

try:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
    FREE_API = os.environ["FREE_API"]
except Exception as e:
    raise ConfigError(str(e))
