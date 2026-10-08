import os

from dotenv import load_dotenv

from .exceptions import ConfigError

load_dotenv()

try:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
    FREE_API = os.environ["FREE_API"]
except KeyError as e:
    raise ConfigError(e.args[0]) from e
