# Deploy ML (FastAPI) to Render Free Tier

This service is the AI pricing + phone search backend (FastAPI).

## Why cold start happens
On Render free tier the service sleeps when idle. First request after sleep may take time because:
- container spins up
- embeddings model loads
- vector DB becomes available

This repo includes optimizations:
- `vectorDB/` is shipped in the Docker image
- embeddings model is pre-downloaded at image build time

## Create Render Web Service (Docker)
1. In Render: **New → Web Service**
2. Connect your GitHub repo
3. Select **Environment: Docker**
4. Set **Root Directory** to `ml`
5. Deploy

## Required env vars
Set these in Render → Service → Environment:
- `GROQ_API_KEY` = your Groq key
- `FORCE_AI_PRICE` = `true` (forces the API to return 503 while warming up instead of returning an estimated price)
- `VECTOR_DB_WAIT_SECONDS` = `20` (how long `/calculate-price` will wait for vector DB readiness before returning 503)

Optional:
- `PORT` is provided by Render automatically

## Frontend config
In the frontend `.env` (Reprise-AI/.env):
- `VITE_AI_API_URL=https://<your-render-ml-service>.onrender.com`

## Expected behavior
- On cold start, `/calculate-price` returns `503` for a few seconds.
- The UI waits and shows "AI warming up… please wait (xxs)" until the AI quote is ready.

If you still see very long warmups, your free tier instance is likely sleeping often; that is expected.
