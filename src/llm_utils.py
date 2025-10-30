from openai import OpenAI
from dotenv import load_dotenv
import os

def get_openai_client():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in .env")
    return OpenAI(api_key=api_key)
