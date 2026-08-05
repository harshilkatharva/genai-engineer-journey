from ..config import GOOGLE_API_KEY
from google import genai


class GoogleProvider:
    def __init__(self):
        self.client = genai.Client(api_key=GOOGLE_API_KEY)

    def get_answer(
        self,
        prompt: str = "",
        model: str = "gemini-3.5-flash-lite",
        config: dict | None = None,
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
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=config if config else None,
            )
            print(response.text)
            return response

        except Exception as e:
            raise e
