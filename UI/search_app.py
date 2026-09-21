import os
import time

import psycopg2
import streamlit as st
import torch
from pgvector.psycopg2 import register_vector
from sentence_transformers import CrossEncoder, SentenceTransformer

# ============================================================
# CONFIGURATION
# ============================================================

TABLE_NAME = "embedding"  # Neon table (was "queue" on localhost)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_POOL = 50
MAX_CANDIDATE_POOL = 200


def get_database_url():
    """Read the Neon connection string from the environment or Streamlit secrets.

    Never hardcode it in this file. Either:
      PowerShell:  $env:DATABASE_URL="postgresql://USER:PASS@HOST/neondb?sslmode=require"
      CMD:         set "DATABASE_URL=postgresql://USER:PASS@HOST/neondb?sslmode=require"
      or create .streamlit/secrets.toml containing:
          DATABASE_URL = "postgresql://USER:PASS@HOST/neondb?sslmode=require"
    (Use quotes exactly as shown; & breaks an unquoted CMD command.)
    """
    url = os.getenv("DATABASE_URL")
    if not url:
        try:
            url = st.secrets["DATABASE_URL"]
        except Exception:
            url = None
    return url


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Semantic Search Engine",
    layout="wide",
)


# ============================================================
# MODELS
# ============================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embed_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.caption(f"Embedding device: `{device}`")
    return SentenceTransformer(EMBED_MODEL_NAME, device=device)


@st.cache_resource(show_spinner="Loading reranker model...")
def get_reranker():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.caption(f"Reranker device: `{device}`")
    return CrossEncoder(RERANK_MODEL_NAME, device=device)


# ============================================================
# DATABASE CONNECTION
# ============================================================

@st.cache_resource
def get_connection():
    """Create the Neon connection and register pgvector."""

    url = get_database_url()
    if not url:
        st.error(
            "DATABASE_URL is not set.\n\n"
            "Set it as an environment variable or in .streamlit/secrets.toml "
            "(see the comment in get_database_url)."
        )
        st.stop()

    conn = psycopg2.connect(url)
    conn.autocommit = True  # no idle open transaction between searches
    register_vector(conn)
    return conn


def run_vector_search(query_embedding, candidate_pool):
    """Run the nearest-neighbour query. Reconnect once if Neon dropped the connection
    (Neon suspends idle compute, which kills a cached connection)."""

    sql = f"""
        SELECT
            text,
            embedding <=> %s AS distance
        FROM {TABLE_NAME}
        ORDER BY embedding <=> %s
        LIMIT %s;
    """

    for attempt in (1, 2):
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, (query_embedding, query_embedding, candidate_pool))
                return cursor.fetchall()
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            get_connection.clear()  # drop the dead connection, reconnect next loop
            if attempt == 2:
                raise


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
):
    """
    Two-stage semantic search:

    1. Generate query embedding.
    2. Retrieve candidate documents using pgvector.
    3. Rerank candidates using CrossEncoder.
    4. Return top-k results.
    """

    if not query.strip():
        return [], {}

    embed_model = get_embed_model()
    reranker = get_reranker()

    timings = {}

    # 1. QUERY EMBEDDING
    start = time.perf_counter()
    query_embedding = embed_model.encode(
        query,
        normalize_embeddings=True,
    ).astype("float32")
    timings["embedding"] = time.perf_counter() - start

    # 2. VECTOR SEARCH
    start = time.perf_counter()
    candidates = run_vector_search(query_embedding, candidate_pool)
    timings["retrieval"] = time.perf_counter() - start

    if not candidates:
        timings["reranking"] = 0.0
        return [], timings

    # 3. CROSS-ENCODER RERANKING
    start = time.perf_counter()
    pairs = [(query, text) for text, _distance in candidates]
    scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    timings["reranking"] = time.perf_counter() - start

    # 4. FINAL RESULTS
    results = []
    for (text, distance), score in reranked[:top_k]:
        results.append(
            {
                "text": text,
                "distance": float(distance),
                "score": float(score),
            }
        )

    return results, timings


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Search Settings")

    top_k = st.slider(
        "Results to show",
        min_value=1,
        max_value=20,
        value=DEFAULT_TOP_K,
    )

    candidate_pool = st.slider(
        "Candidates before reranking",
        min_value=top_k,
        max_value=MAX_CANDIDATE_POOL,
        value=max(DEFAULT_CANDIDATE_POOL, top_k),
    )

    st.divider()

    st.subheader("Models")
    st.caption(f"Embedding model: `{EMBED_MODEL_NAME}`")
    st.caption(f"Reranker: `{RERANK_MODEL_NAME}`")

    st.divider()

    st.subheader("Search Pipeline")

    st.markdown(
        """
        **Query**

        ↓

        **MiniLM Embedding**

        ↓

        **pgvector (Neon)**

        ↓

        **Candidate Retrieval**

        ↓

        **CrossEncoder**

        ↓

        **Final Ranking**
        """
    )


# ============================================================
# MAIN UI
# ============================================================

st.title("Semantic Search Engine")

st.write(
    "Semantic document retrieval using vector embeddings "
    "and CrossEncoder reranking."
)

st.caption(
    f"Table: `{TABLE_NAME}` | "
    f"Candidates: `{candidate_pool}` | "
    f"Results: `{top_k}`"
)


# ============================================================
# SEARCH INPUT
# ============================================================

query = st.text_input(
    "Enter your query",
    placeholder="e.g. Why do children become aggressive?",
)


# ============================================================
# SEARCH
# ============================================================

if query.strip():

    with st.spinner("Searching..."):
        try:
            results, timings = semantic_search(
                query=query,
                top_k=top_k,
                candidate_pool=candidate_pool,
            )
        except Exception as error:
            st.error(f"Search failed: {error}")
            st.stop()

    # PERFORMANCE
    total_time = sum(timings.values())

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Embedding", f"{timings['embedding'] * 1000:.0f} ms")
    with col2:
        st.metric("Retrieval", f"{timings['retrieval'] * 1000:.0f} ms")
    with col3:
        st.metric("Reranking", f"{timings['reranking'] * 1000:.0f} ms")
    with col4:
        st.metric("Total", f"{total_time * 1000:.0f} ms")

    # RESULTS
    st.subheader(f"Top {len(results)} Results")

    if not results:
        st.info("No relevant documents found.")
    else:
        for i, result in enumerate(results, start=1):
            with st.container(border=True):
                st.markdown(f"### Result {i}")
                st.write(result["text"])

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("CrossEncoder Score", f"{result['score']:.3f}")
                with col2:
                    st.metric("Cosine Distance", f"{result['distance']:.4f}")

else:
    st.info("Enter a query above to search the document collection.")