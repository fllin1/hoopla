import os

import pytest
from dotenv import load_dotenv
from google import genai


@pytest.fixture
def gemini_client():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key is None:
        pytest.skip("No GEMINI_API_KEY found in ./.env")

    return genai.Client(api_key=api_key)
