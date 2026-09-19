import os
import time
from contextlib import asynccontextmanager

import psycopg2
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pgvector.psycopg2 import register_vector
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder

# ============================================================
# CONFIGURATION
# ============================================================

DB_HOST = os.getenv("PGHOST", "localhost")
DB_NAME = os.getenv("PGDATABASE", "postgres")
DB_USER = os.getenv("PGUSER", "postgres")
DB_PASSWORD = os.getenv("PGPASSWORD")  # NO hardcoded default. Fail loudly instead.

TABLE_NAME = "queue"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_POOL = 50
MAX_CANDIDATE_POOL = 200

# Env var the Streamlit UI service must set CORS access from.
# e.g. https://your-streamlit-app.koyeb.app
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")

# ============================================================
# GLOBAL STATE (loaded once at startup, not per-request)
# ============================================================

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not DB_PASSWORD:
        raise RuntimeError(
            "PGPASSWORD environment variable is not set. "
            "Set it before starting the service — no default is provided on purpose."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[startup] Using device: {device}")

    print("[startup] Loading embedding model...")
    state["embed_model"] = SentenceTransformer(EMBED_MODEL_NAME, device=device)

    print("[startup] Loading reranker model...")
    state["reranker"] = CrossEncoder(RERANK_MODEL_NAME, device=device)

    print("[startup] Connecting to database...")
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    register_vector(conn)
    state["conn"] = conn

    yield  # app runs here

    print("[shutdown] Closing DB connection...")
    state["conn"].close()


app = FastAPI(title="Semantic Search API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class SearchRequest(BaseModel):
    query: str
    top_k: int = DEFAULT_TOP_K
    candidate_pool: int = DEFAULT_CANDIDATE_POOL


class SearchResult(BaseModel):
    text: str
    distance: float
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]
    timings: dict[str, float]


# ============================================================
# CORE LOGIC (unchanged from your Streamlit script — this part was clean)
# ============================================================

def semantic_search(query: str, top_k: int, candidate_pool: int):
    if not query.strip():
        return [], {}

    embed_model = state["embed_model"]
    reranker = state["reranker"]
    conn = state["conn"]

    timings = {}

    start = time.perf_counter()
    query_embedding = embed_model.encode(
        query, normalize_embeddings=True
    ).astype("float32")
    timings["embedding"] = time.perf_counter() - start

    start = time.perf_counter()
    sql = f"""
        SELECT text, embedding <=> %s AS distance
        FROM {TABLE_NAME}
        ORDER BY embedding <=> %s
        LIMIT %s;
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, (query_embedding, query_embedding, candidate_pool))
            candidates = cursor.fetchall()
    except Exception:
        conn.rollback()
        raise
    timings["retrieval"] = time.perf_counter() - start

    if not candidates:
        return [], timings

    start = time.perf_counter()
    pairs = [(query, text) for text, _distance in candidates]
    scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    timings["reranking"] = time.perf_counter() - start

    results = [
        {"text": text, "distance": float(distance), "score": float(score)}
        for (text, distance), score in reranked[:top_k]
    ]

    return results, timings


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if req.top_k < 1 or req.top_k > 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")
    if req.candidate_pool < req.top_k or req.candidate_pool > MAX_CANDIDATE_POOL:
        raise HTTPException(
            status_code=400,
            detail=f"candidate_pool must be between top_k and {MAX_CANDIDATE_POOL}",
        )

    try:
        results, timings = semantic_search(req.query, req.top_k, req.candidate_pool)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Search failed: {error}")

    return {"results": results, "timings": timings}
