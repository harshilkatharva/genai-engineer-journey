"""

#1

import asyncio


async def retry_async(coro_fn,max_attempt = 3, backoff_seconds=1.0):
    for attempt in range(max_attempt):
        try:
            return await coro_fn
        except Exception:
            if attempt == max_attempt-1:
                raise
            delay = backoff_seconds* (2 ** attempt)
            await asyncio.sleep(delay)

"""

"""
#2

from pydantic import BaseModel, field_validator

class Prediction(BaseModel):
    confidence : float 

    @field_validator("confidence")
    @classmethod

    def validate_confidence(cls, value):
        if not (0 <= value <= 1):
            raise ValueError("confidence must be between 0 and 1")
        return value

"""

# 3

"""

import asyncio 
import time
from functools import wraps

def ratelimiter():
    def __init__(self,rate):
        self.capacity = rate
        self.token = rate
        self.rate = rate
        self.updated = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        while True:
            async with self.lock:
                now = time.monotonic()
                elapsed = now - self.updated
                self.updated = elapsed

            self.token = min(self.capacity,self.token + elapsed * self.rate)

            if self.tokens >= 1:
                self.tokens -= 1
                return

    async def __aenter__(self):
        await self.acquire()

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with self:
                return await func(*args, **kwargs)
        return wrapper


"""


# 4

"""

import asyncio


sem = asyncio.Semaphore(2)

async def employee(name):
    async with sem:
        print(f"{name} started")
        await asyncio.sleep(2)
        print(f"{name} finished")

async def main():
    await asyncio.gather(
        employee("A"), 
        employee("B"),
        employee("C"),
        employee("D")
    )

asyncio.run(main())

"""

"""

#5

import asyncio
import time
import logging

logging.basicConfig(filename="app.log" ,level=logging.INFO)
logger = logging.getLogger(__name__)

class LLM_logger:
    async def __aenter__(self):
        self.start = time.monotonic()
        logger.info("LLM call started")
        return self

    async def __aexit__(self,exc_type,exc,tb):
        end = time.monotonic()
        duration = end - self.start
        logger.info(f"LLM Execution end here in {duration}")
        if exc:
            logger.exception(f"LLM call failed: {exc}")
        return False


async def fake_llm():
    await asyncio.sleep(2)
    return "Done" 

async def main():
    async with LLM_logger():
        response = await fake_llm()
        print(response)

asyncio.run(main())

"""

"""
#6
import asyncio

async def merge_stream(stream_chunk):
    buffer = ""

    async for chunk in stream_chunk:
        buffer += chunk

        if buffer.endswith((".", "!", "?")):
            yield buffer.strip()
            buffer = ""

    if buffer:
        yield buffer.strip()


"""

"""

#7

import unittest
from dataclasses import dataclass

@dataclass
class Token_usages:
    prompt : int
    resposne : int

class TestTokenUsages(unittest.TestCase):
    def test_equal(self):
        token1 = Token_usages(10,20)
        token2 = Token_usages(10,20)

        self.assertEqual(token1, token2)

    def test_not_equal(self):
        token1 = Token_usages(10,20)
        token2 = Token_usages(15,30)

        self.assertNotEqual(token1,token2)

    def test_repr(self):
        token1 = Token_usages(10,20)

        self.assertEqual(repr(token1), "Token_usages(prompt=10, resposne=20)")


if __name__ == "__main__":
    unittest.main()

"""

"""

#8

from functools import wraps
import asyncio

class LLMTimeoutError(Exception):
    pass

def timeout(seconds=5):
    def decorator(func):

        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), seconds)
            except asyncio.TimeoutError:
                raise LLMTimeoutError(f"LLM Exceeded {seconds} seconds")

        return wrapper

    return decorator


import asyncio

@timeout(seconds=3)
async def fake_llm():
    await asyncio.sleep(5)
    return "Done"


async def main():
    try:
        print(await fake_llm())
    except LLMTimeoutError as e:
        print(e)

asyncio.run(main())


"""

"""
#9

from pydantic import BaseModel, Field
from typing import Optional

class ModelDemo(BaseModel):
    name : str
    age : int = Field(ge=0)
    city : str = Optional

def json_converter(model : type[BaseModel]):

    return model.model_json_schema()

print(json_converter(ModelDemo))

"""

"""

#10

import httpx

class HTTPClient:
    def __init__(self):
        self.client = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.client.aclose()

    async def get(self, url, **kwargs):
        respose = await self.client.get(url, **kwargs)

        return respose

    async def post(self, url, **kwargs):
        respose = await self.client.post(url, **kwargs)

        return respose


"""
