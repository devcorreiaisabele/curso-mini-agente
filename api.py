import importlib
import time
import traceback
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google.genai import types
from pydantic import BaseModel

agente = importlib.import_module("04_agente_nutri")

app = FastAPI(title="API Nutri IA")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessoes = {}
MAX_SESSOES = 200


def nova_conversa():
    return agente.cliente.chats.create(
        model=agente.MODELO_CHAT,
        config=types.GenerateContentConfig(
            system_instruction=agente.SYSTEM,
            tools=agente.TOOLS,
            temperature=0.3,
        ),
    )


class Pergunta(BaseModel):
    mensagem: str
    session_id: str | None = None


@app.exception_handler(Exception)
async def erro_global(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"erro": True, "tipo": type(exc).__name__, "detalhe": str(exc)[:500]},
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(p: Pergunta):
    texto = (p.mensagem or "").strip()
    if not texto:
        return {"session_id": p.session_id, "resposta": "Pode me dizer como posso ajudar? 💚", "erro": False}

    sid = p.session_id or str(uuid.uuid4())
    if sid not in sessoes:
        if len(sessoes) >= MAX_SESSOES:
            sessoes.pop(next(iter(sessoes)))
        sessoes[sid] = nova_conversa()

    for i in range(3):
        try:
            resposta = sessoes[sid].send_message(texto).text
            return {
                "session_id": sid,
                "resposta": resposta or "Não consegui responder agora. Pode reformular?",
                "erro": False,
            }
        except Exception:
            traceback.print_exc()
            time.sleep(2 * (i + 1))

    return {
        "session_id": sid,
        "resposta": "Estou com instabilidade agora. Tenta de novo em instantes 💚",
        "erro": True,
    }


@app.delete("/chat/{session_id}")
def reiniciar(session_id: str):
    sessoes.pop(session_id, None)
    return {"ok": True}