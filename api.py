from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from google.genai import errors as genai_errors
from pydantic import BaseModel, Field

from rag_pipeline import carregar_banco, responder_com_rag


class PerguntaRequest(BaseModel):
    pergunta: str = Field(..., min_length=1, max_length=2000)


class Fonte(BaseModel):
    arquivo: str
    pagina: int
    trecho: str


class RespostaResponse(BaseModel):
    resposta: str
    fontes: list[Fonte]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carrega o banco vetorial e o modelo de embeddings uma única vez, na inicialização
    app.state.vector_db = carregar_banco()
    yield
    app.state.vector_db.client.close()


app = FastAPI(title="AI Support Agent", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/perguntar", response_model=RespostaResponse)
def perguntar(req: PerguntaRequest):
    try:
        return responder_com_rag(app.state.vector_db, req.pergunta)
    except genai_errors.ServerError:
        raise HTTPException(status_code=503, detail="Serviço de IA temporariamente indisponível. Tente novamente.")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erro ao gerar resposta: {e}")
