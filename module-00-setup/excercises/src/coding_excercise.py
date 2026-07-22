import os
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
load_dotenv()
import time
from functools import wraps
from pydantic_settings import BaseSettings




#1
def load_required_env(names : list[str]) -> dict:
    """
    Load required environment variables from .env file and return all missing variables at once.
    
    Args:
        names (list[str]): List of environment variable names to load.

    Returns:
        dict: A dictionary containing missing environment variables with their names as keys and 'Missing' as values.
    """

    env_vars = {}
    for name in names:
        value = os.environ.get(name)
        if value is None:
            env_vars[name] = 'Missing'
    return env_vars

missing = load_required_env(['OPENAI_API_KEY', 'GOOGLE_API_KEY', 'ANTHROPIC_API_KEY', 'POSTGRES_URL', 'REDIS_URL', 'TEST_ENV'])
print(f"Missing environment variables: {missing}")

'''

#2
def redact_api_key(api_key: str) -> str:
    """
    Redact an API key by replacing all but first 4 and last 4 characters with asterisks.
    
    Args:
        api_key (str): The API key to redact.
        
    Returns:
        str: The redacted API key.
    """
    if not api_key or len(api_key) <= 4:
        return "api key not found"

    elif len(api_key) <= 8 and len(api_key) > 4:
        return f"{api_key[:2]} {'.' * (len(api_key) - 4)} {api_key[-2:]}"

    else:
        return f"{api_key[:4]} .... {api_key[-4:]}"


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
print(f"Redacted OpenAI API Key: {redact_api_key(OPENAI_API_KEY)}")




#3
def genrate_gitignore_content(project_type : list[str]):
    """"
    Take a project types and write appropriate content in .gitignore.

    Args : 
        project_type (list[str]): List of project types to generate .gitignore content for.

    """

    with open(".gitignore", "w") as gitignore_file:
        gitignore_file.write("# Generated .gitignore file\n\n")
        gitignore_file.write("# Command values \n")
        gitignore_file.write(".env\n")
        gitignore_file.write(".DS_Store\n")
        gitignore_file.write(".vscode/\n")
        gitignore_file.write(".venv/\n\n")

        for project in project_type:
            if project == "python":
                gitignore_file.write("# Python project \n")
                gitignore_file.write("*.pyc\n")
                gitignore_file.write("__pycache__/\n")
            elif project == "node":
                gitignore_file.write("# Node.js project \n")
                gitignore_file.write("node_modules/\n")
                gitignore_file.write(".env\n")
            else:
                print(f"Unsupported project type: {project}. Skipping.")

            # Add more project types as needed

    
genrate_gitignore_content(["python", "java"])

#4

def retry(max_attempts=3, backoff=2):
    """"
    Decorator to retry a function call in case of exceptions, with exponential backoff.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    print(f"Attempt {attempts} failed with error: {e}. Retrying in {backoff ** attempts} seconds...")
                    if attempts >= max_attempts:
                        raise e
                    time.sleep(backoff ** attempts)
        return wrapper
    return decorator



@retry(max_attempts=3, backoff=2)
def call_api(api_client, model_name:str, input_text:str):
    """
    Call the specified API client with the given model name and input text.

    Args:
        api_client: The API client instance to use for the call.
        model_name (str): The name of the model to use.
        input_text (str): The input text to send to the model.

    Returns:
        dict: A dictionary containing the response from the API call.
    """
    try:
        if isinstance(api_client, OpenAI):
            response = api_client.responses.create(
                model=model_name,
                input=input_text
            )
            return {"status": "success", "response": response.output_text}
        elif isinstance(api_client, genai.Client):
            response = api_client.interactions.create(
                model=model_name,
                input=input_text
            )

            return {"status": "success", "response": str(response.output_text)}

        else:
            raise ValueError("Unsupported API client type.")

    except Exception as e:
        raise ValueError(f"API call failed: {e}")
    
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai_client = OpenAI(api_key=OPENAI_API_KEY)
result = call_api(openai_client, "gpt-4.1-mini", "Hello World!")

# GEMINI_API_KEY = os.environ.get('GOOGLE_API_KEY')
# gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# result = call_api(gemini_client, "gemini-3.1-flash-lite", "Hello World!")


print(result)


#5
def sum(a,b):
    return a+b

def measure_latency(function):
    """
    Decorator to measure the latency of a function call.

    Args:
        function: The function to measure latency for.

    Returns:
        The result of the function call along with the latency in seconds.
    """
    @wraps(function)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = function(*args, **kwargs)
        end_time = time.perf_counter()
        latency = (end_time - start_time) / 1000
        return {"result": result, "latency": latency}
    return wrapper

measure_latency_sum = measure_latency(sum)
print(measure_latency_sum(5, 10))  # Example usage of the measure_latency decorator



def genrate_gitignore_content(project_type : list[str]):
    """"
    Take a project types and write appropriate content in .gitignore.

    Args : 
        project_type (list[str]): List of project types to generate .gitignore content for.

    """

    with open(".gitignore", "w") as gitignore_file:
        gitignore_file.write("# Generated .gitignore file\n\n")
        gitignore_file.write("# Command values \n")
        gitignore_file.write(".env\n")
        gitignore_file.write(".DS_Store\n")
        gitignore_file.write(".vscode/\n")
        gitignore_file.write(".venv/\n\n")

        for project in project_type:
            if project == "python":
                gitignore_file.write("# Python project \n")
                gitignore_file.write("*.pyc\n")
                gitignore_file.write("__pycache__/\n")
            elif project == "node":
                gitignore_file.write("# Node.js project \n")
                gitignore_file.write("node_modules/\n")
                gitignore_file.write(".env\n")
            else:
                print(f"Unsupported project type: {project}. Skipping.")

            # Add more project types as needed

    
measure_latency_gitignore = measure_latency(genrate_gitignore_content)
print(measure_latency_gitignore(["python", "java"]))

'''

'''
#6 
def check_openai_api_key_fomat(key:str)->bool:
    """
    Check if the OpenAI API key is in the correct format.

    Returns:
        bool: True if the API key is in the correct format, False otherwise.
    """
    if not key:
        return False
    if not key.startswith("sk-") or len(key) <48 or len(key) >= 170:
        return False
    return True


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if check_openai_api_key_fomat(OPENAI_API_KEY):
    print("OpenAI API key is in the correct format.")
else :
    print("OpenAI API key is NOT in the correct format.")



#7


class config_check(BaseSettings):
    OPENAI_API_KEY: str
    GOOGLE_API_KEY: str
    # ANTHROPIC_API_KEY: str
    REDIS_HOST: str
    REDIS_PORT: int


conf = config_check()

#8

def check_open_port(host:str, port:int)->bool:
    """
    Check if a given port is open on a specified host.

    Args:
        host (str): The hostname or IP address to check.
        port (int): The port number to check.

    Returns:
        bool: True if the port is open, False otherwise.
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)  # Set a timeout for the connection attempt
        result = sock.connect_ex((host, port))
        return result == 0  # Return True if the connection was successful (port is open)

check_redis_port = check_open_port(conf.REDIS_HOST, conf.REDIS_PORT)
print(check_redis_port)


#9

def env_example_cheker():
    """
    Example function to demonstrate checking environment variables.
    """
    with open("../../.env", "r") as env_file:
        env_vars = env_file.readlines()
        env_vars = [var.split('=')[0].strip() for var in env_vars]

    with open("../../.env.example", "r") as env_example_file:
        env_example_vars = env_example_file.readlines()
        env_example_vars = [var.split('=')[0].strip() for var in env_example_vars]

    missing_vars = []
    for var in env_example_vars:
        if var not in env_vars:
            missing_vars.append(var)

    if len(missing_vars) == 0:
        print("Not any env variables missing")
    else:
        print(f"Missing variables :- {missing_vars}")
env_example_cheker()


#10

def call_api(api_client, model_name:str, input_text:str):
    """
    Call the specified API client with the given model name and input text.

    Args:
        api_client: The API client instance to use for the call.
        model_name (str): The name of the model to use.
        input_text (str): The input text to send to the model.

    Returns:
        dict: A dictionary containing the response from the API call.
    """
    try:
        if isinstance(api_client, OpenAI):
            response = api_client.responses.create(
                model=model_name,
                input=input_text
            )
            return {"status": "success", "response": response}
        elif isinstance(api_client, genai.Client):
            response = api_client.interactions.create(
                model=model_name,
                input=input_text
            )

            return {"status": "success", "response": response}

        else:
            raise ValueError("Unsupported API client type.")

    except Exception as e:
        return e

def check_json_response(json):
    """
    Check JSON reposne fail modes and clearly logged failure modes
    """

    if not json:
        return "Json not found that means Network Error"


    status_code = json['status_code']

    if status_code == 200:
        if "error" in json['content'].lower():
            return "malformed json"

    if status_code != 200 and status_code >= 400 and status_code <= 499:
        if status_code == 404:
            return "Showing 404 status code that means reponse not found"
        elif status_code == 401:
            return "Showing 401 status code that means you are unauthorise"
        else:
            return f"Showing {status_code} status_code that means user side problem"

            

'''