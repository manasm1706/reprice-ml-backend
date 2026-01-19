# ML FastAPI service (Render-friendly)
# - Pre-downloads the sentence-transformers model at build time
# - Ships the prebuilt Chroma vectorDB directory in the image
# - Keeps startup non-blocking (app already initializes vector DB in background)

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers

WORKDIR /app

# System deps (kept minimal)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt

# Copy app code + data + vector DB
COPY backend /app/backend
COPY data /app/data
COPY vectorDB /app/vectorDB

# Pre-download embedding model so cold starts are faster on free tier
# (This warms HuggingFace cache inside the image.)
RUN python -c "from backend.tools import get_embeddings; get_embeddings(); print('✅ embeddings warmed')"

# Render provides PORT env var
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
