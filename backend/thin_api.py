import os
import csv
import math
import re
import hashlib
from pathlib import Path
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import uuid

from pinecone import Pinecone

from langchain_core.messages import HumanMessage, SystemMessage

# Load env vars reliably regardless of CWD.
_project_root_env = Path(__file__).resolve().parents[1] / ".env"
if _project_root_env.exists():
    load_dotenv(dotenv_path=_project_root_env)
else:
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


class AdminSearchRequest(BaseModel):
    q: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=50)


class AdminUpsertPhoneRequest(BaseModel):
    id: str | None = None
    brand: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    variant: str | None = None
    price: float | None = None
    image: str | None = None
    link: str | None = None
    metadata: dict[str, Any] | None = None


class AdminUpsertBatchRequest(BaseModel):
    items: list[dict[str, Any]] = Field(..., min_length=1)


class PricingRequest(BaseModel):
    model_name: str
    turns_on: bool
    screen_condition: str
    has_box: bool
    has_bill: bool
    is_under_warranty: bool


def _phones_csv_path() -> Path:
    env = (os.getenv("PHONES_CSV_PATH") or "").strip()
    if env:
        return Path(env)
    # repo layout: reprice-aiback/backend/thin_api.py -> reprice-aiback/data/phones.csv
    return Path(__file__).resolve().parents[1] / "data" / "phones.csv"


def _load_phones_db() -> list[dict[str, Any]]:
    path = _phones_csv_path()
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            rows.append(row)
    return rows


phones_db: list[dict[str, Any]] = _load_phones_db()


def _coerce_finite_float(value) -> float | None:
    try:
        if value is None:
            return None
        num = float(value)
        return num if math.isfinite(num) else None
    except Exception:
        return None


def _estimate_base_price_from_pinecone(model_name: str) -> float | None:
    # Optional: only if Pinecone credentials are configured.
    if not (os.getenv("PINECONE_API_KEY") or "").strip():
        return None

    full_query = (model_name or "").strip()
    if not full_query:
        return None

    if "," in full_query:
        model_query = full_query.split(",")[0].strip()
        user_specs = full_query.split(",", 1)[1].strip().lower()
        user_specs_compact = re.sub(r"\s+", "", user_specs)
    else:
        model_query = full_query
        user_specs_compact = None

    try:
        phones = search_phones(model_query, top_k=8)
        if not phones:
            return None

        def _full_text(md: dict[str, Any]) -> str:
            brand = str(md.get("brand") or "").strip()
            model = str(md.get("model") or "").strip()
            variant = str(md.get("variant") or "").strip()
            return f"{brand} {model} {variant}".strip().lower()

        candidates: list[dict[str, Any]] = [md for md in phones if isinstance(md, dict)]

        # Spec-aware selection (matches the legacy graph node behavior)
        if user_specs_compact:
            for md in candidates:
                text = re.sub(r"\s+", "", _full_text(md))
                if user_specs_compact in text:
                    price = _coerce_finite_float(md.get("price"))
                    if price is not None and price > 0:
                        return float(price)

        # Fall back to the first usable price
        for md in candidates:
            price = _coerce_finite_float(md.get("price"))
            if price is not None and price > 0:
                return float(price)
    except Exception:
        return None

    return None


def _estimate_base_price_from_csv(model_name: str) -> float | None:
    if not phones_db:
        return None

    full_query = (model_name or "").strip()
    if not full_query:
        return None

    if "," in full_query:
        model_query_raw = full_query.split(",")[0].strip()
        user_specs = full_query.split(",", 1)[1].strip().lower()
    else:
        model_query_raw = full_query
        user_specs = None

    def _norm_spaces(s: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split())

    def _compact(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

    model_query = _norm_spaces(model_query_raw)
    if not model_query:
        return None

    query_tokens = set(model_query.split())
    spec_compact = _compact(user_specs) if user_specs else None

    best_price: float | None = None
    best_score: float = -1.0

    for p in phones_db:
        brand = (p.get("brand", "") or "").strip()
        model = (p.get("model", "") or "").strip()
        variant = (p.get("variant", "") or "").strip()

        full_name_raw = f"{brand} {model} {variant}".strip()
        full_name = _norm_spaces(full_name_raw)
        if not full_name:
            continue

        full_compact = _compact(full_name_raw)
        full_tokens = set(full_name.split())

        score = 0.0
        if model_query in full_name:
            score = 100.0
        else:
            if query_tokens:
                overlap = len(query_tokens & full_tokens) / max(1, len(query_tokens))
                score = overlap * 10.0

        if spec_compact and spec_compact in full_compact:
            score += 5.0

        if score <= best_score:
            continue

        price = _coerce_finite_float(p.get("price"))
        if price is None or price <= 0:
            continue

        best_price = float(price)
        best_score = score

    if best_price is None or best_score < 2.0:
        return None

    return best_price


def _estimate_base_price(model_name: str) -> tuple[float | None, str]:
    pc_price = _estimate_base_price_from_pinecone(model_name)
    if pc_price is not None and pc_price > 0:
        return pc_price, "pinecone"

    csv_price = _estimate_base_price_from_csv(model_name)
    if csv_price is not None and csv_price > 0:
        return csv_price, "csv"

    return None, "none"


def extract_decimal_from_text(text: str) -> float:
    """Robustly parse a deduction percentage from LLM output (graph.py compatibility)."""
    try:
        match = re.search(r"0\.\d+|1\.0|\.\d+", text or "")
        if match:
            return float(match.group())
        match_int = re.search(r"\d+", text or "")
        if match_int:
            val = float(match_int.group())
            return val / 100 if val > 1 else val
        return 0.0
    except Exception:
        return 0.0


@lru_cache(maxsize=512)
def get_dynamic_deduction(model_name: str, issue_type: str) -> float:
    """AI-based deduction factor (0..1). Falls back to the same hardcoded values used in graph.py."""
    issue = (issue_type or "").strip()
    if not issue or issue.lower() == "good":
        return 0.0

    # If Groq isn't configured, use deterministic fallbacks.
    if not (os.getenv("GROQ_API_KEY") or "").strip():
        if "shattered" in issue.lower():
            return 0.60
        if "cracked" in issue.lower():
            return 0.35
        if "major" in issue.lower():
            return 0.20
        if "minor" in issue.lower():
            return 0.10
        return 0.0

    prompt = f"""You are an expert mobile phone appraiser specializing in the Indian resale market.

Phone Model: {model_name}
Screen Issue: {issue}

Provide the market deduction percentage (0.0 to 1.0) for this defect based on these guidelines:

SEVERITY SCALE:
• Minor Scratches: 0.05-0.10 (light surface marks, fully functional)
• Major Scratches: 0.15-0.20 (deep/multiple scratches, visible in use)
• Cracked: 0.25-0.35 (hairline/corner cracks, touch works)
• Shattered: 0.50-0.70 (spiderweb cracks, glass missing, display damaged)

ADJUSTMENT FACTORS:
1. Flagship/Premium phones (iPhone Pro, Samsung Ultra, Fold/Flip): Use HIGHER end due to expensive screen replacement costs
2. Mid-range phones: Use middle of range
3. Budget phones (<₹15k original price): Use LOWER end unless shattered

Return ONLY a decimal number (e.g., 0.35). DO NOT write any text or reasoning.""".strip()

    try:
        msg = _llm().invoke(
            [
                SystemMessage(content="You output only a number between 0.0 and 1.0."),
                HumanMessage(content=prompt),
            ]
        )

        content = getattr(msg, "content", str(msg)).strip()
        deduction = extract_decimal_from_text(content)

        # Sanity check like graph.py
        if deduction == 0.0 and issue in ["Cracked", "Shattered"]:
            raise ValueError("AI returned 0.0 for severe damage")

        # Clamp to [0,1]
        return max(0.0, min(1.0, float(deduction)))
    except Exception:
        if "shattered" in issue.lower():
            return 0.60
        if "cracked" in issue.lower():
            return 0.35
        if "major" in issue.lower():
            return 0.20
        if "minor" in issue.lower():
            return 0.10
        return 0.0


def _quick_estimated_price(request: PricingRequest) -> dict[str, Any]:
    base_price, source = _estimate_base_price(request.model_name)
    logs: list[str] = []

    if base_price is None or base_price <= 0:
        logs.append("⚠️ No base price match found. Returning ₹0.")
        return {"final_price": 0.0, "base_price": None, "logs": logs, "estimated": True, "source": source}

    if not request.turns_on:
        logs.append("❌ Mobile does not turn on. Price set to ₹0.")
        return {"final_price": 0.0, "base_price": float(base_price), "logs": logs, "estimated": True, "source": source}

    calculated_price = float(base_price)

    # Screen condition deduction (graph.py compatibility)
    condition = (request.screen_condition or "Good").strip() or "Good"
    if condition != "Good":
        deduction_factor = get_dynamic_deduction(request.model_name, condition)
        deduction_amount = calculated_price * deduction_factor
        calculated_price -= deduction_amount
        logs.append(
            f"📉 Condition '{condition}': applied {int(deduction_factor * 100)}% deduction (-₹{deduction_amount:.2f})."
        )
    else:
        logs.append("✅ Screen Condition: Good (No deduction)")

    # Box/Bill/Warranty adjustments (same as graph.py)
    if not request.has_bill:
        calculated_price -= 1000
        logs.append("📄 Missing Bill: -₹1,000")

    if not request.has_box:
        calculated_price -= 500
        logs.append("📦 Missing Box: -₹500")

    if request.is_under_warranty:
        bonus = calculated_price * 0.10
        calculated_price += bonus
        logs.append(f"🛡️ Under Warranty: +10% Bonus (+₹{bonus:.2f})")

    logs.insert(0, f"ℹ️ Pricing computed (source: {source}).")
    return {
        "final_price": max(0.0, calculated_price),
        "base_price": float(base_price),
        "logs": logs,
        # Mark as not-estimated when we have a usable base price; UI can still treat it as a quote.
        "estimated": False,
        "source": source,
    }


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


def _embed_passage(text: str) -> list[float]:
    out = _pc().inference.embed(
        model=_pinecone_embed_model(),
        inputs=[text],
        parameters={"input_type": "passage"},
    )
    return out[0]["values"]


def _embed_passages(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    out = _pc().inference.embed(
        model=_pinecone_embed_model(),
        inputs=texts,
        parameters={"input_type": "passage"},
    )
    return [item["values"] for item in out]


def _phone_text(brand: str, model: str, variant: str | None) -> str:
    parts = [brand.strip(), model.strip()]
    if variant and variant.strip():
        parts.append(variant.strip())
    return " ".join([p for p in parts if p])


def _clean_metadata(md: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for k, v in md.items():
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        clean[k] = v
    return clean


def _stable_vector_id(brand: str, model: str, variant: str | None) -> str:
    b = (brand or "").strip()
    m = (model or "").strip()
    v = (variant or "").strip()
    base = f"{b}|{m}|{v}".encode("utf-8")
    digest = hashlib.sha1(base).hexdigest()[:16]
    return f"phone_{digest}"


def _normalize_phone_row(row: dict[str, Any]) -> dict[str, Any]:
    # Accept flexible column names; treat unknown keys as extra metadata.
    raw = row or {}
    if not isinstance(raw, dict):
        raise ValueError("Row must be an object")

    def g(*keys: str) -> Any:
        for k in keys:
            if k in raw:
                return raw.get(k)
            lk = k.lower()
            for rk, rv in raw.items():
                if isinstance(rk, str) and rk.lower() == lk:
                    return rv
        return None

    brand = str(g("brand", "Brand") or "").strip()
    model = str(g("model", "Model") or "").strip()
    variant_val = g("variant", "Variant")
    variant = str(variant_val).strip() if variant_val is not None else None
    variant = variant if variant else None

    if not brand or not model:
        raise ValueError("Missing required brand/model")

    id_val = g("id", "ID")
    vid = str(id_val).strip() if id_val is not None else ""
    if not vid:
        vid = _stable_vector_id(brand, model, variant)

    price = _coerce_finite_float(g("price", "Price"))

    image = g("image", "Image")
    link = g("link", "Link", "url", "URL")
    image_s = str(image).strip() if image is not None else ""
    link_s = str(link).strip() if link is not None else ""
    if not image_s and link_s and link_s.startswith("http"):
        # Many CSVs use "link" as an image URL.
        image_s = link_s

    known = {"id", "brand", "model", "variant", "price", "image", "link", "url"}
    extras: dict[str, Any] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            continue
        if k.lower() in known:
            continue
        extras[k] = v

    base_md: dict[str, Any] = {
        "brand": brand,
        "model": model,
        "variant": variant,
        "price": price,
        "image": image_s or None,
        "link": link_s or None,
    }

    md = _clean_metadata({**base_md, **extras})
    text = _phone_text(brand, model, variant)
    return {"id": vid, "text": text, "metadata": md}


def search_phones(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    vector = _embed_query(query)
    resp = _index().query(vector=vector, top_k=top_k, include_metadata=True)

    phones: list[dict[str, Any]] = []
    for match in _coerce_matches(resp):
        md = match.get("metadata") if isinstance(match, dict) else getattr(match, "metadata", None)
        if isinstance(md, dict) and md:
            phones.append(md)

    return phones


@app.post("/admin/phones/search")
def admin_search(req: AdminSearchRequest):
    try:
        vector = _embed_query(req.q)
        resp = _index().query(vector=vector, top_k=req.top_k, include_metadata=True)

        matches_out: list[dict[str, Any]] = []
        for match in _coerce_matches(resp):
            if isinstance(match, dict):
                mid = match.get("id")
                score = match.get("score")
                md = match.get("metadata")
            else:
                mid = getattr(match, "id", None)
                score = getattr(match, "score", None)
                md = getattr(match, "metadata", None)

            if not mid or not isinstance(md, dict):
                continue
            matches_out.append({"id": str(mid), "score": score, "metadata": md})

        return {"query": req.q, "count": len(matches_out), "matches": matches_out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/phones/upsert")
def admin_upsert(req: AdminUpsertPhoneRequest):
    try:
        vid = (req.id or "").strip() or str(uuid.uuid4())

        base_md: dict[str, Any] = {
            "brand": req.brand.strip(),
            "model": req.model.strip(),
            "variant": (req.variant.strip() if req.variant else None),
            "price": req.price,
            "image": req.image,
            "link": req.link,
        }

        extra = req.metadata or {}
        if not isinstance(extra, dict):
            extra = {}
        md = _clean_metadata({**base_md, **extra})

        text = _phone_text(req.brand, req.model, req.variant)
        values = _embed_passage(text)

        _index().upsert([{"id": vid, "values": values, "metadata": md}])
        return {"ok": True, "id": vid, "message": "Upserted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/phones/upsert-batch")
def admin_upsert_batch(req: AdminUpsertBatchRequest):
    try:
        # Requires Pinecone credentials to embed and upsert.
        _require_env("PINECONE_API_KEY")

        normalized: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for idx, row in enumerate(req.items):
            try:
                normalized.append(_normalize_phone_row(row))
            except Exception as e:
                errors.append({"row": idx + 1, "error": str(e)})

        if not normalized:
            return {"ok": False, "upserted": 0, "failed": len(errors), "errors": errors}

        # Embed + upsert in chunks to keep requests small.
        chunk_size = 100
        upserted = 0

        for start in range(0, len(normalized), chunk_size):
            chunk = normalized[start : start + chunk_size]
            texts = [str(item["text"]) for item in chunk]
            embeddings = _embed_passages(texts)

            vectors = []
            for item, values in zip(chunk, embeddings):
                vectors.append({"id": str(item["id"]), "values": values, "metadata": item["metadata"]})

            _index().upsert(vectors)
            upserted += len(vectors)

        return {
            "ok": len(errors) == 0,
            "upserted": upserted,
            "failed": len(errors),
            "errors": errors,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/admin/phones/{phone_id}")
def admin_delete(phone_id: str):
    try:
        pid = phone_id.strip()
        if not pid:
            raise HTTPException(status_code=400, detail="Missing id")
        _index().delete(ids=[pid])
        return {"ok": True, "id": pid, "message": "Deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@app.post("/calculate-price")
def calculate_price(req: PricingRequest):
    try:
        return _quick_estimated_price(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
