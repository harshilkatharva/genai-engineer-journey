import sys
from pathlib import Path

from google import genai

from config import GOOGLE_API_KEY
from utils.experiment_logger import experiment_logger

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


google_client = genai.Client(api_key=GOOGLE_API_KEY)


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
    else:
        experiment_logger.error(f"No response received for prompt: {prompt}")
        return "No response received."


# 1
"""
sentences = {
    "english": "The quick brown fox jumps over the lazy dog while the sun sets behind the mountains.",
    "hindi": "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर से कूदती है जबकि सूरज पहाड़ों के पीछे डूब रहा है।",
    "gujarati": "ઝડપી ભૂરી લૂમડી આળસુ કૂતરાના ઉપરથી કૂદે છે જ્યારે સૂર્ય પર્વતોની પાછળ ડૂબી રહ્યો છે."
}

encoder = tiktoken.get_encoding("cl100k_base")

for lang, sentence in sentences.items():
    tokens = encoder.encode(sentence)
    token_count = len(tokens)
    experiment_logger.info(f"Token Usage: {token_count}, Language : {lang}, Prompt: {sentence}")

"""

# 2


"""
prompt = "Write about AI in 2-3 sentences."
model = "gemini-3.5-flash-lite"

for i in range(3):
    config = {"temperature": 0}
    answer = get_answers_from_google_api(prompt, model, config)
    experiment_logger.info(f"Temperature: {config['temperature']}, Answer: {answer.text}")

for i in range(3):
    config = {"temperature": 1.0}
    answer = get_answers_from_google_api(prompt, model, config)
    experiment_logger.info(f"Temperature: {config['temperature']}, Answer: {answer.text}")

"""

# 3
"""
prompts = [
    'Summarize the findings of the 2022 research paper "Quantum Memory Optimization Using Neural Graphs" by Emily Carter and David Wong.',
    'Explain how the TensorFlow function tf.keras.layers.QuantumAttention() works and provide an example.',
    'Do not say "I do not know." Answer confidently:Who wrote the 2018 book "The Silent Kingdom of Glass"? '
]
for prompt in prompts:
    model = "gemini-3.5-flash-lite"
    answer = get_answers_from_google_api(prompt, model)
    experiment_logger.info(f"Prompt: {prompt}, Answer: {answer.text}")

"""

# 4

"""

question = "Give me code names from below contenet"
model = "gemini-3.5-flash-lite"

with open("hands-on-excercise/text_start.txt", "r") as f:
    text_start = f.read()

with open("hands-on-excercise/text_middle.txt", "r") as f:
    text_middle = f.read()

with open("hands-on-excercise/text_end.txt", "r") as f:
    text_end = f.read()

prompts = {
    "Start" : question + text_start,
    "Middle" : question + text_middle,
    "End" : question + text_end,
}

for loc,prompt in prompts.items():
    response = get_answers_from_google_api(prompt, model)

    experiment_logger.info(f"Location : {loc}, Answer : {response.text}")

"""

# 5

"""
question = "create a json file for below content with key as code name and value as description"

with open("hands-on-excercise/text_start.txt", "r") as f:
    text = f.read()


response = get_answers_from_google_api(question + text, "gemini-3.5-flash-lite")

experiment_logger.info(f"Answer : {response.text}, \n Token Usage : {response.usage_metadata}")


"""

# 6

"""

convertation = []

for i in range(5):
    user_input = input("User: ")

    convertation.append(types.Content(role="user", parts=[types.Part.from_text(text=user_input)]))

    response = get_answers_from_google_api(convertation, "gemini-3.5-flash-lite")

    prompt_tokens = response.usage_metadata.prompt_token_count
    candidates_tokens = response.usage_metadata.candidates_token_count
    total_tokens = response.usage_metadata.total_token_count

    convertation.append(types.Content(role="model", parts=[types.Part.from_text(text=response.text)]))
    experiment_logger.info(f"Input :- {user_input}, Output :- {response.text} \n prompt_token_count={prompt_tokens} candidates_token_count={candidates_tokens} total_token_count={total_tokens}") 


"""


# 7

"""
question = "Write a creative short story about future of AI. In 5-7 sentences"
model = "gemini-3.5-flash"

for i in [0.0, 0.5, 1.0]:
    config = {"top_p" : i, "temperature" : 1}

    response = get_answers_from_google_api(question, model, config)

    experiment_logger.info(f"Top_p : {i},Temperature : 1, Answer : {response.text}, \n Token Usage : {response.usage_metadata.total_token_count}, Model : {model}")

"""
