from rich.console import Console
from rich.table import Table
import config
from openai import OpenAI
from anthropic import Anthropic
from google import genai
import time


console = Console()

# Fetching API Key from config
OPENAI_API_KEY = getattr(config, 'OPENAI_API_KEY', None)
GOOGLE_API_KEY = getattr(config, 'GOOGLE_API_KEY', None)
ANTHROPIC_API_KEY = getattr(config, 'ANTHROPIC_API_KEY', None)
POSTGRESQL_CONNECTION = getattr(config, 'POSTGRESQL_CONNECTION', False)
REDIS_CONNECTION = getattr(config, 'REDIS_CONNECTION', False)

# Schema of result
results = {
    'OpenAI' : {'key_available' : 'Yes' if OPENAI_API_KEY else 'No', 
                'latency' : None,
                'result' : None
    },
    'Google' : {'key_available' : 'Yes' if GOOGLE_API_KEY else 'No', 
                'latency' : None,
                'result' : None
    },
    'Anthropic' : {'key_available' : 'Yes' if ANTHROPIC_API_KEY else 'No', 
                'latency' : None,
                'result' : None
    },
}

with console.status("[bold blue]Checking API's...", spinner="dots"):
    # OpenAI Check
    if OPENAI_API_KEY:
        start = time.perf_counter()
        try:
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            response = openai_client.responses.create(
                model='gpt-4.1-mini',
                input="Hello World!"
            )
            end = time.perf_counter()
            results['OpenAI']['latency'] = end - start
            results['OpenAI']['results'] = 'success'
            
        except Exception as e:
            end = time.perf_counter()
            results['OpenAI']['latency'] = end - start
            results['OpenAI']['result'] = 'failed'

     # Google Check
    if GOOGLE_API_KEY:
        start = time.perf_counter()
        try:
            google_client = genai.Client(api_key=GOOGLE_API_KEY)
            response = google_client.interactions.create(
                model="gemini-3.6-flash",
                input="Hello World!"
            )
            end = time.perf_counter()
            results['Google']['latency'] = end - start
            results['Google']['result'] = 'success'
            
        except Exception as e:
            end = time.perf_counter()
            results['Google']['latency'] = end - start
            results['Google']['result'] = 'failed'

    # Anthropic Check
    if ANTHROPIC_API_KEY:
        start = time.perf_counter()
        try:
            anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
            response = anthropic_client.messages.create(
                model="claude-3.0",
                messages = [
                    {
                        'role' : 'user',
                        'content' : 'Hello World!'
                    }
                ]
            )
            end = time.perf_counter()
            results['Anthropic']['latency'] = end - start
            results['Anthropic']['result'] = 'success'
        except Exception as e:
            end = time.perf_counter()
            results['Anthropic']['latency'] = end - start
            results['Anthropic']['result'] = 'failed'

# Create table for showing result of our API's
api_table = Table(title="Health Check Results for API's")
api_table.add_column("Provider", style="cyan", no_wrap=True)
api_table.add_column("API Key Available", style="magenta")
api_table.add_column("Latency (seconds)", style="green")
api_table.add_column("Result", style="yellow")

for provider, data in results.items():
    api_table.add_row(
        provider,
        data['key_available'],
        str(data['latency']) if data['latency'] is not None else "N/A",
        data['result'] if data['result'] is not None else "N/A"
    )


console.print(api_table)

database_table = Table(title="Health Check Results for Databases")
database_table.add_column("Database", style="cyan", no_wrap=True)
database_table.add_column("Connection Status", style="magenta")

database_table.add_row("PostgreSQL",str(POSTGRESQL_CONNECTION))
database_table.add_row("Redis",str(REDIS_CONNECTION))

console.print(database_table)