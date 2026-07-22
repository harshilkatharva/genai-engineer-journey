from config import OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY
from openai import OpenAI
from anthropic import Anthropic
from google import genai

print("Calling OpenAI...")
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    model_list = openai_client.models.list()
    print(f"Available OpenAI models: {[model.id for model in model_list.data]}")
    print("OpenAI call successful!")

except Exception as e:
    print(f"OpenAI call failed: {e}")


print("\nCalling Google...")
try:
    google_client = genai.Client(api_key=GOOGLE_API_KEY)
    models = google_client.models.list()
    print(f"Available Google models: {[model.name for model in models]}")
    print("Google call successful!")

except Exception as e:
    print(f"Google call failed: {e}")


print("\nCalling Anthropic...")
try:
    anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    models = anthropic_client.models.list()
    print(f"Available Anthropic models: {[model.id for model in models.data]}")
    print("Anthropic call successful!")
except Exception as e:
    print(f"Anthropic call failed: {e}")
