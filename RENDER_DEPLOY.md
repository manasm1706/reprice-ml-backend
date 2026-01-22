# Deploy FastAPI (Thin RAG) to Render Free Tier

This service is a thin API:

FastAPI (Render)
	→ Pinecone (vector search + embeddings)
	→ Groq (LLM reasoning)

No local vector DB, no Chroma, no sentence-transformers, no torch.

## Create Render Web Service (Docker)
1. In Render: **New → Web Service**
2. Connect your GitHub repo
3. Select **Environment: Docker**
4. Set **Root Directory** to `reprice-aiback` (or the repo root if this is the repo)
5. Deploy

## Required env vars
Set these in Render → Service → Environment:
- `PINECONE_API_KEY`
- `PINECONE_INDEX` (example: `reprice-index`)
- `GROQ_API_KEY`

Optional:
- `PINECONE_EMBED_MODEL` (default: `llama-text-embed-v2`)
- `GROQ_MODEL` (default: `llama-3.1-70b-versatile`)
- `PORT` is provided by Render automatically

## Endpoints
- `GET /health`
- `POST /search` body: `{ "q": "...", "top_k": 5 }`
- `POST /ask` body: `{ "q": "...", "top_k": 5 }`
