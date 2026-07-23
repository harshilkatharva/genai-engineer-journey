import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
ANTHROPIC_API_KEY = os.environ['ANTHROPIC_API_KEY']

