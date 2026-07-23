'''
#1
import httpx
import asyncio
import time

test_api = 'https://jsonplaceholder.typicode.com/todos/1'


def test_sequential(api) -> float:
    start = time.perf_counter()
    for i in range(0,3):
        response = httpx.get(api)
        response.json()
        i += 1
    end = time.perf_counter()


    return end - start 


sequential_time = test_sequential(test_api)
print(sequential_time)

async def test_concurrently(api) -> float:
    start = time.perf_counter()
    async with httpx.AsyncClient() as client:
        calls = [client.get(api) for _ in range(0,3)]
        responses = await asyncio.gather(*calls)

        for resposne in responses:
            resposne.json()

    return time.perf_counter() - start

concurrent_time = asyncio.run(test_concurrently(test_api))
print(concurrent_time)

'''

'''
#2
from pydantic import BaseModel, Field
from typing import Literal


class product_review(BaseModel):
    rating : int = Field(ge=1, le=5)
    sentiment : Literal['Positive', 'Negative', 'Neutral']
    complaints = list[str]
'''

'''

#3
from typing import Iterator
import asyncio

async def fake_stream_genrator(text):
    tokens = text.split()
    for token in tokens:
        await asyncio.sleep(2)
        yield token + " "

async def test_stream():
    async for token in fake_stream_genrator("Hello, How are you!"):
        print(token, end="", flush=True)

asyncio.run(test_stream())


'''

'''
#4

class LLMError(Exception):
    pass

class LLMRateLimitError(LLMError):
    pass

class LLMTimeoutError(LLMError):
    pass

class LLMContentFilterError(LLMError):
    pass

class LLMResposneError(LLMError):
    pass


def call_llm(error_type : str):
    if error_type == "rate_limit":
        raise LLMRateLimitError("429 Too many Requests")

    elif error_type == "timeout":
        raise LLMTimeoutError("Request Timeout")

    elif error_type == "content_filter":
        raise LLMContentFilterError("Blocked by Content Filter")

    elif error_type == "invalid":
        raise LLMResposneError("Invalid JSON response")
    else:
        raise LLMError("Invalid")


try :
    call_llm("rate_limitvfdfdbv")

except LLMRateLimitError:
    print("Retry after few seconds")

except LLMTimeoutError:
    print("Retry after few seconds")

except LLMContentFilterError:
    print("Modify the prompt content is inappropriate")

except LLMResposneError:
    print("Log error and inspect responses")

except LLMError:
    print("Unknown LLM Error")

'''


from pydantic import BaseModel
import asyncio
import httpx

test_api = 'https://jsonplaceholder.typicode.com/todos/1'

def check_response(BaseModel):
    userId : int
    id : int
    title : str
    body : str

async def call_api(api):
    async with httpx.AsyncClient as client:
        response = await client.get(api)
        response.raise_for_status() 
        return response
    # check_response(response)
    

result = asyncio.run(call_api(test_api))
print(result)

