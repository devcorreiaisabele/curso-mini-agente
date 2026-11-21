import os
import chromadb
from pypdf import PdfReader

# 1. LER O ARQUIVO
diretorio_base = os.path.dirname(os.path.abspath(__file__))
caminho_pdf = os.path.join(diretorio_base, "faq_projeto.pdf")

leitor = PdfReader(caminho_pdf)
texto = ""
for pagina in leitor.pages:
    texto += (pagina.extract_text() or "") + "\n"

# 2. CHUNKS COM OVERLAP (JANELA DESLIZANTE)
tamanho_chunk = 450
overlap = 80
chunks = []
inicio = 0

while inicio < len(texto):
    chunks.append(texto[inicio : inicio + tamanho_chunk])
    inicio += tamanho_chunk - overlap

# 3. BANCO VETORIAL (CHROMADB COM EMBEDDING NATIVO)
caminho_banco = os.path.join(diretorio_base, "banco_vetorial_faq")

cliente_chroma = chromadb.PersistentClient(path=caminho_banco)
"""
get_or_create_collection usa o embedding nativo do ChromaDB
(rápido e eficiente, sem necessidade de PyTorch)
"""
colecao = cliente_chroma.get_or_create_collection(
    name="faq_colecao",
    metadata={"hnsw:space": "cosine"}
)

ids = [f"chunk_{i}" for i in range(len(chunks))]
colecao.upsert(ids=ids, documents=chunks)
print(f"{len(chunks)} chunks salvos com sucesso!")

# 4. TESTE DE BUSCA SEMÂNTICA
pergunta = "Qual é a política de garantia e devolução?"
resultado = colecao.query(
    query_texts=[pergunta], n_results=1
)
print("\n--- Teste de Busca ---")
print("Pergunta:", pergunta)
print("Trecho mais relevante encontrado:")
print(resultado["documents"][0][0].strip())