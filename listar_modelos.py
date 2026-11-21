import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
cliente = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(api_version="v1beta"),
)

for m in cliente.models.list():
    print(m.name)