import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

chave_api = os.getenv("GEMINI_API_KEY")
cliente = genai.Client(
    api_key=chave_api,
    http_options=types.HttpOptions(api_version="v1beta"),
)
modelo = os.getenv("GEMINI_CHAT_MODEL", "gemini-flash-lite-latest")

print("=" * 60)
print("CHAMADA 1 — o usuário se apresenta")
print("=" * 60)

apresentacao = (
    "Oi! Meu nome é Ana\n"
    "e meu projeto é um sistema\n"
    "de estoque de uma farmácia."
)
resposta_1 = cliente.models.generate_content(
    model=modelo,
    contents=[
        {"role": "user", "parts": [{"text": apresentacao}]},
    ],
)

texto_resposta_1 = resposta_1.text
print("USER     :", apresentacao)
print("ASSISTANT:", texto_resposta_1)