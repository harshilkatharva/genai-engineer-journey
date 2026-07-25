# import pytest

# from ..client import LLMClient


# @pytest.mark.asyncio
# async def test_complete_all():

#     client = LLMClient()

#     results = await client.complete_all(
#         "Explain AI"
#     )

#     assert len(results) == 3

#     providers = {
#         result.provider
#         for result in results
#     }

#     assert providers == {
#         "openai",
#         "anthropic",
#         "google",
#     }