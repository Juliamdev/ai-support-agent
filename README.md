# AI Support Agent

Agente de suporte com **RAG** (Retrieval-Augmented Generation): indexa documentos PDF em um banco vetorial (Qdrant) e responde perguntas usando o Google Gemini, **apenas** com base no conteúdo dos documentos e indicando as páginas usadas como fonte.

## Stack

- **LLM:** Google Gemini (SDK `google-genai`), com modelo reserva caso o principal esteja indisponível
- **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` (local, multilíngue, sem custo de API)
- **Banco vetorial:** Qdrant (local em disco ou servidor)
- **API:** FastAPI + Uvicorn
- **Orquestração:** LangChain (loader, splitter, vector store)

## Instalação

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env       # e preencha a GEMINI_API_KEY
```

## Uso

**1. Indexar o documento** (só precisa rodar de novo quando o PDF mudar):

```bash
python rag_pipeline.py --indexar data/manual.pdf
```

**2a. Perguntar pelo terminal:**

```bash
python rag_pipeline.py
```

**2b. Ou subir a API:**

```bash
uvicorn api:app --reload
```

Documentação interativa em http://localhost:8000/docs.

```bash
curl -X POST http://localhost:8000/perguntar \
  -H "Content-Type: application/json" \
  -d '{"pergunta": "Qual é a formação acadêmica?"}'
```

Resposta:

```json
{
  "resposta": "...",
  "fontes": [{ "arquivo": "manual.pdf", "pagina": 1, "trecho": "..." }]
}
```

## Qdrant: local ou servidor

- **Local (padrão):** com `QDRANT_URL` vazio, os dados ficam na pasta `qdrant_db/`. Nesse modo só um processo acessa o banco por vez, então pare a API antes de reindexar.
- **Servidor:** suba com `docker run -p 6333:6333 qdrant/qdrant` e defina `QDRANT_URL=http://localhost:6333`.

## Configuração (`.env`)

| Variável | Padrão | Descrição |
|---|---|---|
| `GEMINI_API_KEY` | — | Chave da API do Gemini (obrigatória) |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Modelo principal |
| `GEMINI_FALLBACK_MODEL` | `gemini-flash-latest` | Modelo usado se o principal estiver indisponível |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Modelo de embeddings |
| `QDRANT_URL` | vazio | URL do servidor Qdrant (vazio = local) |
| `QDRANT_PATH` | `qdrant_db` | Pasta do banco local |
| `COLLECTION_NAME` | `suporte_docs` | Nome da coleção |
| `TOP_K` | `3` | Quantidade de trechos recuperados por pergunta |
