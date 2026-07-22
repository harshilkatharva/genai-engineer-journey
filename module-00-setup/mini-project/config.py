import os
from dotenv import load_dotenv
from rich import print
from rich.console import Console
import psycopg
import redis

load_dotenv()

console = Console()

console.rule('[bold blue]Checking API KEY Availability')

try:
    OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
    print("[green]OPENAI_API_KEY found in environment variables.[/green]")
except KeyError:
    print("[red]OPENAI_API_KEY is missing in environment variables. Please set it in the .env file.[/red]")

try:
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
    print("[green]GOOGLE_API_KEY found in environment variables.[/green]")
except KeyError:
    print("[red]GOOGLE_API_KEY is missing in environment variables. Please set it in the .env file.[/red]")

try:
    ANTHROPIC_API_KEY = os.environ['ANTHROPIC_API_KEY']
    print("[green]ANTHROPIC_API_KEY found in environment variables.[/green]")
except KeyError:
    print("[red]ANTHROPIC_API_KEY is missing in environment variables. Please set it in the .env file.[/red]")

console.rule('[green] Finished API KEY Availability')

console.rule('[bold blue]Checking Database Connections')

# PostgreSQL Check
try:
    conn = psycopg.connect(
        dbname=os.environ.get('POSTGRES_DB'), 
        user=os.environ.get('POSTGRES_USER'),
        password=os.environ.get('POSTGRES_PASSWORD'),
        host=os.environ.get('POSTGRES_HOST'),
        port=os.environ.get('POSTGRES_PORT')
    )
    conn.close()
    POSTGRESQL_CONNECTION = True
    console.print("[green]PostgreSQL connection successful![/green]")
except Exception as e:
    POSTGRESQL_CONNECTION = False
    console.print(f"[red]PostgreSQL connection failed: {e}[/red]")

# Redis Check
try:
    conn = redis.Redis(host=os.environ.get('REDIS_HOST'), port=os.environ.get('REDIS_PORT'))
    conn.ping()
    REDIS_CONNECTION = True
    console.print("[green]Redis connection successful![/green]")
except Exception as e:
    REDIS_CONNECTION = False
    console.print(f"[red]Redis connection failed: {e}[/red]")

console.rule('[green] Finished Database Connections')

