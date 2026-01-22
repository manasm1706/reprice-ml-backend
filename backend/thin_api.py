import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pinecone import Pinecone

from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing {name}")
    return value


def _pinecone_index_name() -> str:
    return os.getenv("PINECONE_INDEX", "reprice-index")


def _pinecone_embed_model() -> str:
    return os.getenv("PINECONE_EMBED_MODEL", "llama-text-embed-v2")


def _groq_model() -> str:
    return os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")


@lru_cache(maxsize=1)
def _pc() -> Pinecone:
    return Pinecone(api_key=_require_env("PINECONE_API_KEY"))


@lru_cache(maxsize=1)
def _index():
    return _pc().Index(_pinecone_index_name())


@lru_cache(maxsize=1)
def _llm():
    from langchain_groq import ChatGroq

    return ChatGroq(api_key=_require_env("GROQ_API_KEY"), model=_groq_model(), temperature=0.2)

app = FastAPI(title="Reprice AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    q: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)


class AskRequest(BaseModel):
    q: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)


def _coerce_matches(resp: Any) -> list[Any]:
    if resp is None:
        return []

    if isinstance(resp, dict):
        return list(resp.get("matches") or [])

    matches = getattr(resp, "matches", None)
    if matches is not None:
        return list(matches)

    return []


def _embed_query(text: str) -> list[float]:
    out = _pc().inference.embed(
        model=_pinecone_embed_model(),
        inputs=[text],
        parameters={"input_type": "query"},
    )
    return out[0]["values"]


def search_phones(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    vector = _embed_query(query)
    resp = _index().query(vector=vector, top_k=top_k, include_metadata=True)

    phones: list[dict[str, Any]] = []
    for match in _coerce_matches(resp):
        md = match.get("metadata") if isinstance(match, dict) else getattr(match, "metadata", None)
        if isinstance(md, dict) and md:
            phones.append(md)

    return phones


@app.get("/health")
def health():
    return {"ok": True, "index": _pinecone_index_name()}


@app.post("/search")
def search(req: SearchRequest):
    try:
        phones = search_phones(req.q, req.top_k)
        return {"query": req.q, "count": len(phones), "phones": phones}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask(req: AskRequest):
    try:
        phones = search_phones(req.q, req.top_k)

        prompt = f"""
User wants: {req.q}

Available phones (metadata JSON list):
{phones}

Task:
- Recommend the best option(s) for the user.
- Be concise and practical.
- If nothing matches well, say so and ask exactly 1 clarifying question.
""".strip()

        msg = _llm().invoke(
            [
                SystemMessage(content="You help users choose a phone from a catalog."),
                HumanMessage(content=prompt),
            ]
        )

        return {
            "query": req.q,
            "matches": phones,
            "answer": getattr(msg, "content", str(msg)),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
