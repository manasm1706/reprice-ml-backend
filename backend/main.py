'''
LEGACY IMPLEMENTATION (disabled):
- Loaded CSV locally
- Built/loaded local vector DB (Chroma)
- Ran local embedding models
- Used langgraph

This file is kept only to preserve the entrypoint `backend.main:app`.
The real production/free-tier backend is now in `backend/thin_api.py`.
'''

from backend.thin_api import app

r'''  # Everything below is intentionally disabled.

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.graph import app as pricing_agent
from backend.tools import init_vector_store
import asyncio
import requests
from fastapi.responses import Response
import csv
import os
import re
import math
from typing import List, Dict

# Initialize FastAPI
app = FastAPI(title="RePrice AI API")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Vector DB init tracking
vector_db_ready_event: asyncio.Event = asyncio.Event()
vector_db_init_task: asyncio.Task | None = None
vector_db_init_error: str | None = None

# Load Phone Data from CSV
phones_db = []
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(BASE_DIR, "data", "phones.csv")

    if os.path.exists(csv_path):
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'link' in row and row.get('link'):
                    row['image'] = row.get('link')
                elif 'image' not in row:
                    row['image'] = ''
                phones_db.append(row)
        print(f"✅ Loaded {len(phones_db)} phones from CSV")
    else:
        print(f"⚠️ CSV file not found at {csv_path}")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")

@app.get("/")
def home():
    """Health check endpoint - always responds quickly"""
    return {
        "message": "RePrice AI API is Running",
        "vector_db_ready": vector_db_ready_event.is_set(),
        "vector_db_initializing": (vector_db_init_task is not None and not vector_db_init_task.done()),
        "vector_db_error": vector_db_init_error,
        "phones_loaded": len(phones_db)
    }

@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "vector_db_ready": vector_db_ready_event.is_set(),
        "vector_db_initializing": (vector_db_init_task is not None and not vector_db_init_task.done()),
        "vector_db_error": vector_db_init_error,
        "phones_count": len(phones_db)
    }

@app.on_event("startup")
async def startup_event():
    """Initialize vector DB in background without blocking startup"""
    global vector_db_init_task, vector_db_init_error
    
    async def init_db():
        global vector_db_init_error
        try:
            print("🚀 Starting vector DB initialization...")
            # Run the blocking operation in a thread pool
            await asyncio.to_thread(init_vector_store)
            vector_db_init_error = None
            vector_db_ready_event.set()
            print("✅ Vector DB ready")
        except Exception as e:
            print(f"❌ Vector DB initialization failed: {e}")
            vector_db_init_error = str(e)
            # keep event unset
    
    # Start initialization in background
    vector_db_init_task = asyncio.create_task(init_db())
    print("🟢 Server started - Vector DB loading in background")


async def _ensure_vector_db_ready(timeout_seconds: float = 25.0) -> bool:
    """Wait for vector DB readiness (best-effort) to avoid transient 503s."""
    if vector_db_ready_event.is_set():
        return True

    # If init failed, don't wait forever.
    if vector_db_init_error:
        return False

    try:
        await asyncio.wait_for(vector_db_ready_event.wait(), timeout=timeout_seconds)
        return True
    except asyncio.TimeoutError:
        return False


def _estimate_base_price_from_csv(model_name: str) -> float | None:
    """Fast best-effort base price from CSV (no embeddings / vector search)."""
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
        # Lowercase, keep alphanumerics, collapse whitespace.
        return " ".join(re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split())

    def _compact(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

    model_query = _norm_spaces(model_query_raw)
    if not model_query:
        return None

    query_tokens = set(model_query.split())
    spec_compact = _compact(user_specs) if user_specs else None

    best: dict | None = None
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

        # Base score: strong boost for substring match after normalization.
        score = 0.0
        if model_query in full_name:
            score = 100.0
        else:
            # Token overlap fallback (helps when punctuation/order differs).
            if query_tokens:
                overlap = len(query_tokens & full_tokens) / max(1, len(query_tokens))
                score = overlap * 10.0

        # Spec bonus if provided (e.g. "6/128", "128gb")
        if spec_compact and spec_compact in full_compact:
            score += 5.0

        if score <= best_score:
            continue

        try:
            price = float(p.get("price", 0) or 0)
        except Exception:
            price = 0.0

        best = {"full_text": full_name, "price": price}
        best_score = score

    # Require at least some overlap to avoid random matches.
    if best is None or best_score < 2.0:
        return None

    return float(best["price"])


def _quick_estimated_price(request: "PricingRequest") -> dict:
    """Return a fast estimated price without vector DB / LLM calls."""
    base_price = _estimate_base_price_from_csv(request.model_name)
    logs: list[str] = []

    if base_price is None or base_price <= 0:
        logs.append("⚠️ No base price match found in CSV. Returning ₹0 estimate.")
        return {"final_price": 0.0, "base_price": None, "logs": logs, "estimated": True}

    if not request.turns_on:
        logs.append("❌ Mobile does not turn on. Price set to ₹0.")
        return {"final_price": 0.0, "base_price": base_price, "logs": logs, "estimated": True}

    # Deterministic condition deductions (same as graph fallbacks)
    condition = request.screen_condition
    deduction_map = {
        "Good": 0.0,
        "Minor Scratches": 0.10,
        "Major Scratches": 0.20,
        "Cracked": 0.35,
        "Shattered": 0.60,
    }
    deduction_factor = float(deduction_map.get(condition, 0.0))

    calculated_price = float(base_price)
    if deduction_factor > 0:
        deduction_amount = calculated_price * deduction_factor
        calculated_price -= deduction_amount
        logs.append(f"📉 Condition '{condition}': applied {int(deduction_factor * 100)}% deduction (-₹{deduction_amount:.2f}).")
    else:
        logs.append("✅ Screen Condition: Good (No deduction)")

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

    logs.insert(0, "ℹ️ Returning estimated pricing (fallback mode).")
    return {
        "final_price": max(0.0, calculated_price),
        "base_price": base_price,
        "logs": logs,
        "estimated": True,
    }


def _coerce_finite_float(value) -> float | None:
    try:
        if value is None:
            return None
        num = float(value)
        return num if math.isfinite(num) else None
    except Exception:
        return None

@app.get("/search-phones")
def search_phones(q: str = Query(..., min_length=1)):
    """Search phones from CSV data"""
    query = q.lower()
    results = []
    
    for p in phones_db:
        brand = p.get('brand', '').strip()
        model = p.get('model', '').strip()
        full_name = f"{brand} {model}".lower()
        
        if query in full_name:
            results.append({
                "brand": brand,
                "model": model,
                "variant": p.get('variant', ''),
                "price": float(p.get('price', 0) or 0),
                "image": p.get('image', '') or p.get('link', '')
            })
            
    return results[:50]

@app.get("/proxy-image")
def proxy_image(url: str):
    """Proxy images to avoid CORS issues"""
    try:
        r = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )
        return Response(
            content=r.content,
            media_type=r.headers.get("Content-Type", "image/jpeg")
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail="Image fetch failed")

class PricingRequest(BaseModel):
    model_name: str
    turns_on: bool
    screen_condition: str
    has_box: bool
    has_bill: bool
    is_under_warranty: bool
    prefer_ai: bool | None = None

@app.post("/calculate-price")
async def calculate_price(request: PricingRequest):
    """
    Calculate phone price using LangGraph Agent
    """
    # Warmup behavior:
    # - Wait a bit (<= ~10s) so users can get the real AI quote without instant fallback.
    # - If still warming up, return 503 so the frontend can retry.
    # - If initialization failed, fall back to deterministic estimate.
    force_ai = os.getenv("FORCE_AI_PRICE", "0").strip().lower() in {"1", "true", "yes"}
    prefer_ai = bool(request.prefer_ai) if request.prefer_ai is not None else force_ai

    wait_seconds_raw = os.getenv("VECTOR_DB_WAIT_SECONDS", "10")
    try:
        wait_seconds = float(wait_seconds_raw)
    except Exception:
        wait_seconds = 10.0

    ready = await _ensure_vector_db_ready(timeout_seconds=max(0.0, wait_seconds))
    if not ready:
        # If user prefers AI, return 503 so the client can wait/retry.
        if prefer_ai:
            raise HTTPException(status_code=503, detail="AI service is warming up. Please retry in a few seconds.")

        # Otherwise: fall back to deterministic estimate (legacy behavior)
        if vector_db_init_error:
            return _quick_estimated_price(request)
        return _quick_estimated_price(request)
    
    # Prepare inputs
    inputs = {
        "model_name": request.model_name,
        "turns_on": request.turns_on,
        "screen_condition": request.screen_condition,
        "has_box": request.has_box,
        "has_bill": request.has_bill,
        "is_under_warranty": request.is_under_warranty,
        "log": [] 
    }
    
    try:
        result = pricing_agent.invoke(inputs)

        final_price = _coerce_finite_float(result.get("final_price"))
        base_price = _coerce_finite_float(result.get("base_price"))
        logs = result.get("log")
        logs_list: list[str] = logs if isinstance(logs, list) else []

        # If the agent couldn't resolve a usable base price:
        # - If forcing AI: return a 404-like error so UI can try a different candidate.
        # - Else: fall back to CSV-based estimate.
        if base_price is None or base_price <= 0:
            if prefer_ai:
                raise HTTPException(status_code=422, detail="AI could not resolve a base price for this model")
            fallback = _quick_estimated_price(request)
            fallback_logs = fallback.get("logs") if isinstance(fallback.get("logs"), list) else []
            merged_logs = []
            merged_logs.extend([l for l in logs_list if isinstance(l, str)])
            merged_logs.append("ℹ️ AI lookup returned no base price; used CSV fallback.")
            merged_logs.extend([l for l in fallback_logs if isinstance(l, str)])
            return {
                "final_price": fallback.get("final_price"),
                "base_price": fallback.get("base_price"),
                "logs": merged_logs,
                "estimated": True,
            }

        # If final price is missing but base price exists:
        # - If forcing AI: return error.
        # - Else: compute fallback final deterministically.
        if final_price is None:
            if prefer_ai:
                raise HTTPException(status_code=422, detail="AI could not resolve a final price")
            fallback = _quick_estimated_price(request)
            fallback_logs = fallback.get("logs") if isinstance(fallback.get("logs"), list) else []
            merged_logs = []
            merged_logs.extend([l for l in logs_list if isinstance(l, str)])
            merged_logs.append("ℹ️ AI lookup returned no final price; computed fallback final price.")
            merged_logs.extend([l for l in fallback_logs if isinstance(l, str)])
            return {
                "final_price": fallback.get("final_price"),
                "base_price": base_price,
                "logs": merged_logs,
                "estimated": True,
            }

        return {
            "final_price": final_price,
            "base_price": base_price,
            "logs": [l for l in logs_list if isinstance(l, str)],
            "estimated": False,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

'''