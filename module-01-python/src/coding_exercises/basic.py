'''

import asyncio
import httpx
from google import genai

async def fetch_completion(client : httpx.AsyncClient, prompt : str) -> str:
    response = await client.interaction.create(
            "https://api.example.com/v1/complete",
            json={"prompt" : prompt},
            timeout = 30.0
    )

    response.raise_for_status()
    return response.json()["text"]

async def main():
    prompts = ['Hello how are you', 'What should you do tell me in 1 line', 'What are you doing']

    async with httpx.AsyncClient as client:

        results_sequential = []
        for prompt in prompts:
            results.append(await fetch_completion(client, prompt))

        result_concerent = await asyncio.gather(*(fetch_completion(client, prompt) for prompt in prompts))


asyncio.run(main())


'''

'''

from pydantic import BaseModel,Field
from typing import Literal


class check_sentiment(BaseModel):
    sentiment : Literal['Positive', 'Negative', 'Neutral'],
    confidence_score : float = Field(ge=0.0, le=1.0)
    key_phrase = list[str]


'''


# from typing import Iterator

# def fake_token_genrator(text:str) -> Iterator[str]:
#     for word in text.split():
#         yield  word + " "


# def async_stream_example(prompt):
#     async def stream_completion(prompt: str):
#         async for chunk in llm_client.stream(prompt):
#             yield chunk.textt


