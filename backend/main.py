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

@app.post("/calculate-price")
async def calculate_price(request: PricingRequest):
    """
    Calculate phone price using LangGraph Agent
    """
    # Avoid returning 503 during warmup: wait briefly for init to complete.
    ready = await _ensure_vector_db_ready(timeout_seconds=25.0)
    if not ready:
        detail = vector_db_init_error or "Vector database is still initializing. Please try again in a few seconds."
        raise HTTPException(status_code=503, detail=detail)
    
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
        
        return {
            "final_price": result.get("final_price"),
            "base_price": result.get("base_price"),
            "logs": result.get("log")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))