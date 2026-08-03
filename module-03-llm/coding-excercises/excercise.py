import tiktoken
from google import genai
import os
from dotenv import load_dotenv
from difflib import ndiff
import json
import matplotlib.pyplot as plt
import time


load_dotenv()  # Load environment variables from .env file


def token_estimate(sentence: str) -> int:
    """
    Estimates the number of tokens in a given sentence using the tiktoken library.

    Args:
        sentence (str): The input sentence for which the token count is to be estimated.

    Returns:
        int: The estimated number of tokens in the input sentence.
    """
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(sentence)
    return len(tokens)


def manage_history(
    convertation_history: list[dict], system_message_token_size: int, max_tokens: int
) -> list[dict]:
    """
    Manages the conversation history to ensure that the total token count does not exceed the specified maximum tokens.

    Args:
        convertation_history (list[dict]): The conversation history, where each entry is a dictionary containing 'role' and 'content'.
        system_message_token_size (int): The token size of the system message.
        max_tokens (int): The maximum allowed tokens for the conversation history.

    Returns:
        list[dict]: The updated conversation history that fits within the specified token limit.
    """

    # we can use 50% of the max tokens for the input prompt and 50% for the output response
    input_token_limit = max_tokens / 2
    manage_history_tokens = system_message_token_size

    final_history = []
    for history in reversed(convertation_history):
        history_token_usage = token_estimate(history["content"])
        if history_token_usage + system_message_token_size <= input_token_limit:
            final_history.append(history)
            manage_history_tokens += history_token_usage
        else:
            break

        return list(reversed(final_history))


def cost_estimation(token_count: int, pricing_table: dict, provider: str) -> int:
    """
    Estimates the cost of using a language model based on the token count and pricing table.

    Args:
        token_count (int): The number of tokens used in the request.
        pricing_table (dict): A dictionary containing the pricing information for different providers.
        provider (str): The name of the provider for which the cost is to be estimated.

    Returns:
        int: The estimated cost for the given token count and provider.
    """
    if provider not in pricing_table:
        raise ValueError(f"Provider '{provider}' not found in pricing table.")

    price_per_token = pricing_table[provider]
    estimated_cost = token_count * price_per_token
    return estimated_cost


google_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


def get_answers_from_google_api(
    prompt: str, model: str, config: dict | None = None
) -> str:
    """
    Function to get answers from Google API using the provided prompt.

    Args:
        prompt (str): The input prompt for which the answer is to be generated.
        model (str): The model to use for generating the answer.
        config (dict | None): The configuration for generating the answer.

    Returns:
        str: The generated answer from the Google API.
    """
    response = google_client.models.generate_content(
        model=model, contents=prompt, config=config if config else None
    )

    if response:
        return response


temps = [0.0, 0.2, 0.5, 0.8, 1.0]
responses = {}
prompt = "Explain machine learning in 1 sentence."

for temp in temps:
    response = get_answers_from_google_api(
        prompt, model="gemini-3.5-flash-lite", config={"temperature": temp}
    )
    responses[temp] = response.text

for i in range(len(temps) - 1):
    t1 = temps[i]
    t2 = temps[i + 1]

    print("=" * 80)
    print(f"Comparing {t1} vs {t2}")

    diff = ndiff(responses[t1].split(), responses[t2].split())

    print("\n".join(diff))


def check_json(output: str) -> str:
    try:
        json.loads(output)
        return "Valid JSON"
    except ValueError as e:
        if (
            "Expecting value" in e.msg
            or "Unterminated string" in e.msg
            or output.split().endswith((",", "{", "[", ":"))
        ):
            return "Truncated JSON"
        else:
            return "Malformed JSON"


def budget_allocation(
    total_token_limit: int,
    system_token_ration: float,
    conversation_token_ratio: float,
    retrieve_token_ration: float,
) -> dict:
    """
    Allocates the total token limit into different categories based on the provided ratios.

    Args:
        total_token_limit (int): The total token limit to be allocated.
        system_token_ration (float): The ratio of tokens to be allocated for system messages.
        conversation_token_ratio (float): The ratio of tokens to be allocated for conversation history.
        retrieve_token_ration (float): The ratio of tokens to be allocated for retrieval.

    Returns:
        dict: A dictionary containing the allocated token limits for each category.
    """
    if not (
        0 <= system_token_ration <= 1
        and 0 <= conversation_token_ratio <= 1
        and 0 <= retrieve_token_ration <= 1
    ):
        raise ValueError("Ratios must be between 0 and 1.")
    if system_token_ration + conversation_token_ratio + retrieve_token_ration != 1:
        raise ValueError("The sum of the ratios must equal 1.")

    system_token_limit = int(total_token_limit * system_token_ration)
    conversation_token_limit = int(total_token_limit * conversation_token_ratio)
    retrieve_token_limit = int(total_token_limit * retrieve_token_ration)

    return {
        "system_token_limit": system_token_limit,
        "conversation_token_limit": conversation_token_limit,
        "retrieve_token_limit": retrieve_token_limit,
    }


def token_bound_segment(chunks: list[str], token_limit: int) -> list[str]:
    """
    Segments a list of text chunks into smaller segments based on a specified token limit.

    Args:
        chunks (list[str]): A list of text chunks to be segmented.
        token_limit (int): The maximum number of tokens allowed in each segment.

    Returns:
        list[str]: A list of segmented text chunks, each within the specified token limit.
    """
    segmented_chunks = []
    token_usages = 0

    for chunk in chunks:
        chunk_token_usage = token_estimate(chunk)

        if token_usages + chunk_token_usage <= token_limit:
            segmented_chunks.append(chunk)
            token_usages += chunk_token_usage
        else:
            break

    return segmented_chunks


def check_hallucination(output: str, expected_answer: str) -> bool:
    """
    Checks if the output from the model contains hallucinations by comparing it with the expected answer.

    Args:
        output (str): The output generated by the model.
        expected_answer (str): The expected correct answer.

    Returns:
        bool: True if hallucination is detected (output does not match expected answer), False otherwise.
    """
    return True if expected_answer in output else False


def compare_latency_by_max_tokens(prompt: str, max_tokens: list[int]) -> None:
    """
    Print relation between latency and max_token for same prompt

    Args:
        prompt : Prompt for comparison.
        max_token : Maximum token limit for that prompt

    Return :
        None but plot comparison
    """

    comparison_data = []

    for token_limit in max_tokens:
        start_time = time.perf_counter()
        response = get_answers_from_google_api(
            prompt,
            model="gemini-3.5-flash-lite",
            config={"max_output_tokens": token_limit},
        )
        if response:
            end_time = time.perf_counter()
            latency = (end_time - start_time) * 100
            comparison_data.append({"max_tokens": token_limit, "latency": latency})

    # Plot the comparison
    plt.figure(figsize=(10, 6))
    plt.plot(
        [data["max_tokens"] for data in comparison_data],
        [data["latency"] for data in comparison_data],
        marker="o",
    )
    plt.xlabel("Max Tokens")
    plt.ylabel("Latency (miliseconds)")
    plt.title("Latency vs Max Tokens")
    plt.show()


def model_evalution(prompt: str, models: dict[str, int]) -> list:
    """
    Evalute model by latency, cost and ouput and print it comparison

    Args:
        prompt : prompt for ask that model
        models : dictinory have name and cost per token of that model
    """

    result = []

    for model_name, cost_per_token in models.items():
        start_time = time.perf_counter()
        response = get_answers_from_google_api(prompt, model_name)
        end_time = time.perf_counter()
        latency = (end_time - start_time) * 100
        total_token_usage = response.usage_metadata.total_token_count
        cost = total_token_usage * cost_per_token
        output_text = response.text

        result.append(
            {
                "Model": model_name,
                "latency": latency,
                "cost": cost,
                "output": output_text,
            }
        )

    return result
