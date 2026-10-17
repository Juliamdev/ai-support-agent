import os
import re
import sys
from functools import lru_cache

from dotenv import load_dotenv

# Imports para Leitura e Busca Vetorial
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

# SDK Oficial do Google Gemini
from google import genai
from google.genai import errors as genai_errors

# 1. Carregar variáveis de ambiente
load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-flash-latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
QDRANT_URL = os.getenv("QDRANT_URL")  # Se vazio, usa o Qdrant local em disco
QDRANT_PATH = os.getenv("QDRANT_PATH", "qdrant_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "suporte_docs")
TOP_K = int(os.getenv("TOP_K", "3"))

MENSAGEM_SEM_RESPOSTA = "Desculpe, não encontrei essa informação nos documentos disponibilizados."


@lru_cache
def get_gemini_client() -> genai.Client:
    """Cria o cliente do Gemini apenas quando for usado pela primeira vez."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("A chave GEMINI_API_KEY não foi encontrada no arquivo .env!")
    return genai.Client(api_key=api_key)


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    """Carrega o modelo de embeddings local (multilíngue) uma única vez."""
    print(f"🔄 Carregando modelo de embeddings local ({EMBEDDING_MODEL})...")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_qdrant_client() -> QdrantClient:
    """Conecta ao servidor Qdrant (QDRANT_URL) ou ao banco local em disco."""
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=os.getenv("QDRANT_API_KEY"))
    return QdrantClient(path=QDRANT_PATH)


def normalizar_texto(texto: str) -> str:
    """Corrige linhas com letras espaçadas (ex: 'J U L I A  M A R Q U E S'),
    comuns em PDFs exportados de ferramentas de design."""
    linhas = []
    for linha in texto.splitlines():
        tokens = linha.split()
        letras_soltas = sum(len(t) == 1 for t in tokens)
        if len(tokens) > 3 and letras_soltas / len(tokens) > 0.8:
            # Palavras separadas por 2+ espaços, letras por 1 espaço
            palavras = re.split(r" {2,}", linha.strip())
            linha = " ".join(p.replace(" ", "") for p in palavras)
        linhas.append(linha)
    return "\n".join(linhas)


def indexar_pdf(pdf_path: str) -> QdrantVectorStore:
    """Lê o PDF, divide em chunks e salva no Qdrant (substitui a coleção existente)."""
    print(f"📖 Lendo o arquivo: {pdf_path}...")
    reader = PdfReader(pdf_path)
    documentos = [
        Document(
            page_content=normalizar_texto(pagina.extract_text() or ""),
            metadata={"source": pdf_path, "page": i},
        )
        for i, pagina in enumerate(reader.pages)
    ]

    print("✂️ Dividindo o documento em trechos (chunks)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documentos)
    print(f"✅ Documento dividido em {len(chunks)} chunks.")

    print("🧠 Gerando embeddings e armazenando no Qdrant...")
    conexao = {"url": QDRANT_URL} if QDRANT_URL else {"path": QDRANT_PATH}
    vector_db = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        force_recreate=True,
        api_key=os.getenv("QDRANT_API_KEY"),
        **conexao,
    )
    print("⚡ Banco de dados vetorial pronto!")
    return vector_db


def carregar_banco() -> QdrantVectorStore:
    """Abre uma coleção já indexada, sem reprocessar o PDF."""
    client = get_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        client.close()
        raise RuntimeError(
            f"A coleção '{COLLECTION_NAME}' não existe. Rode primeiro: python rag_pipeline.py --indexar"
        )
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=get_embeddings(),
    )


def responder_com_rag(vector_db: QdrantVectorStore, pergunta: str) -> dict:
    """Busca contexto no Qdrant e envia ao Gemini. Retorna a resposta e as fontes usadas."""
    # 1. Buscar os trechos mais relevantes do PDF
    docs = vector_db.similarity_search(pergunta, k=TOP_K)
    contexto = "\n\n".join(doc.page_content for doc in docs)

    # 2. Montar o Prompt com as instruções estritas
    prompt_completo = f"""Você é um assistente de suporte especializado.
Responda à pergunta do usuário usando APENAS o contexto fornecido abaixo.
Se você não souber a resposta ou se ela não estiver no contexto, responda honestamente dizendo:
'{MENSAGEM_SEM_RESPOSTA}' Não invente nada.

Contexto dos documentos:
{contexto}

Pergunta: {pergunta}"""

    # 3. Gerar a resposta (tenta o modelo reserva se o principal estiver indisponível)
    response = None
    for modelo in dict.fromkeys([GEMINI_MODEL, GEMINI_FALLBACK_MODEL]):
        try:
            response = get_gemini_client().models.generate_content(
                model=modelo,
                contents=prompt_completo,
            )
            break
        except genai_errors.ServerError:
            if modelo == GEMINI_FALLBACK_MODEL:
                raise
            print(f"⚠️ Modelo {modelo} indisponível, tentando {GEMINI_FALLBACK_MODEL}...")

    fontes = [
        {
            "arquivo": os.path.basename(doc.metadata.get("source", "")),
            "pagina": doc.metadata.get("page", 0) + 1,
            "trecho": doc.page_content[:200],
        }
        for doc in docs
    ]
    return {"resposta": response.text, "fontes": fontes}


# --- EXECUÇÃO VIA TERMINAL ---
# python rag_pipeline.py --indexar [caminho.pdf]  -> indexa o PDF
# python rag_pipeline.py                          -> modo perguntas
if __name__ == "__main__":
    if "--indexar" in sys.argv:
        args = [a for a in sys.argv[1:] if a != "--indexar"]
        caminho_pdf = args[0] if args else "data/manual.pdf"
        if not os.path.exists(caminho_pdf):
            sys.exit(f"❌ Erro: arquivo '{caminho_pdf}' não encontrado.")
        indexar_pdf(caminho_pdf)
        sys.exit(0)

    try:
        vector_db = carregar_banco()
    except RuntimeError as e:
        sys.exit(f"❌ {e}")

    print("\n" + "="*50)
    print("Digite suas perguntas (ou 'sair' para encerrar).")
    while True:
        pergunta = input("\n❓ Pergunta: ").strip()
        if pergunta.lower() in {"sair", "exit", "quit", ""}:
            break

        print("🔍 Buscando contexto e gerando resposta com Gemini...")
        try:
            resultado = responder_com_rag(vector_db, pergunta)
            print("\n🤖 Resposta da IA:")
            print(resultado["resposta"])
            paginas = sorted({f["pagina"] for f in resultado["fontes"]})
            print(f"📄 Fontes: página(s) {', '.join(map(str, paginas))}")
        except Exception as e:
            print(f"\n❌ Ocorreu um erro ao chamar a IA: {e}")
    print("="*50)
