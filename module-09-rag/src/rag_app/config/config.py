from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

DATABASE_CONNECTION_CONVERSATION_URL = os.environ["DATABASE_CONNECTION_CONVERSATION_URL"]

X_API_KEY_Tenant_A = os.environ["X_API_KEY_Tenant_A"]
X_API_KEY_Tenant_B = os.environ["X_API_KEY_Tenant_B"]
