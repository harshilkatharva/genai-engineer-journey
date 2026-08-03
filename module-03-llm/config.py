import os

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
