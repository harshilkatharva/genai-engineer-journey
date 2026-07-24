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

'''
#5

from pydantic import BaseModel
import asyncio
import httpx

test_api = 'https://jsonplaceholder.typicode.com/posts/1'

class check_response(BaseModel):
    userId : int
    id : int 
    title : str
    body : str

async def call_api(api):
    async with httpx.AsyncClient() as client:
        response = await client.get(api)
        # response.raise_for_status() 
        data = response.json()
        try:
            check_response.model_validate(data)
            print(data)
        except Exception as e:
            print(f"Output diffrent from expactations. check error \n {e}")
    

asyncio.run(call_api(test_api))

'''


'''

#6

from unittest.mock import patch,AsyncMock, MagicMock
import httpx
import asyncio
import pytest

test_api = 'https://jsonplaceholder.typicode.com/posts/1'

async def call_api(api):
    async with httpx.AsyncClient() as client:
        response = await client.get(api)
        response.raise_for_status()
        return response.json()

# fake_response = MagicMock()
# fake_response.json.return_value = {
#     'userId': 1, 
#     'id': 1, 
#     'title': 'sunt aut facere repellat provident occaecati excepturi optio reprehenderit', 
#     'body': 'quia et suscipit '
# }

# fake_response.raise_for_status.return_value = None

# with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=fake_response)):
#     result = asyncio.run(call_api(test_api))

# print(result)

@pytest.mark.asyncio
@patch("httpx.AsyncClient.get")
async def fake_call_api(api,mock_get):
    fake_response = MagicMock()
    fake_response.json.return_value = {
        'userId': 1, 
        'id': 1, 
        'title': 'sunt aut facere repellat provident occaecati excepturi optio reprehenderit', 
        'body': 'quia et suscipit '
    }
    fake_response.raise_for_status.return_value = None

    mock_get.return_value = fake_response

    result = await call_api(api)

    return result


print(asyncio.run(fake_call_api(test_api)))
    
'''



# '''

#7

import hashlib
import time
import asyncio

def compute_hash():
    for _ in range(500000):
        hashlib.sha256(b"Hello").hexdigest()

async def cpu_task():
    start = time.perf_counter()
    compute_hash()
    print("CPU-bond time :- ", time.perf_counter() - start)

async def async_test():

    start = time.perf_counter()

    await asyncio.gather(
        cpu_task(),
        cpu_task(),
        cpu_task()
    )

    print("Async task :- ", time.perf_counter() - start)

asyncio.run(async_test())


from concurrent.futures import ProcessPoolExecutor

def worker(_):
    compute_hash()


start = time.perf_counter()

with ProcessPoolExecutor() as executor:
    list(executor.map(worker, range(3)))

print("Preprocess Pool task :- ", time.perf_counter() - start)

# '''

