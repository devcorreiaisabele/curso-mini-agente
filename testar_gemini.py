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

pergunta = "O que é CIência de Dados?"

print("=" * 60)
print("\nCHAMADA 1 — sem instrução de 'system' (comportamento padrão)")

historico = [{"role": "user", "content": pergunta}]

print("USER:", pergunta)

resposta = cliente.chat.completions.create(
    model=modelo,
    messages=historico,
)

print("ASSISTANT:", resposta.choices[0].message.content)
print("=" * 60)
print("=" * 60)
resposta_1 = cliente.chat.completions.create(
    model=modelo,
    messages=[
        {"role": "user", "content": pergunta},
    ]
)
print(resposta_1.choices[0].message.content)

print("\nCHAMADA 2 - Com instrução do 'system'")

sistema = "Você é um professor explicando para uma criança de 8 anos. Responda em no máximo 2 frases simples, com uma analogia do dia a dia."

pergunta_2 = "O que é Ciência de Dados?"

historico = [
    {"role": "system", "content": sistema},
    {"role": "user", "content": pergunta_2}
]
print("SYSTEM:", sistema)
print("USER:", pergunta_2)

resposta_2 = cliente.chat.completions.create(
    model=modelo,
    messages=historico,
)
print("ASSISTANT:", resposta_2.choices[0].message.content)