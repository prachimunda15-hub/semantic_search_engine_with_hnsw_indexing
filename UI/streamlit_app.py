import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL")  # set this in Koyeb for this service, no fallback

DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_POOL = 50
MAX_CANDIDATE_POOL = 200

st.set_page_config(page_title="Semantic Search Engine", layout="wide")

if not API_URL:
    st.error("API_URL environment variable is not set. Point it at your deployed API service.")
    st.stop()

with st.sidebar:
    st.header("Search Settings")
    top_k = st.slider("Results to show", 1, 20, DEFAULT_TOP_K)
    candidate_pool = st.slider(
        "Candidates before reranking", top_k, MAX_CANDIDATE_POOL,
        max(DEFAULT_CANDIDATE_POOL, top_k),
    )

st.title("Semantic Search Engine")
st.write("Semantic document retrieval using vector embeddings and CrossEncoder reranking.")

query = st.text_input("Enter your query", placeholder="e.g. Why do children become aggressive?")

if query.strip():
    with st.spinner("Searching..."):
        try:
            resp = requests.post(
                f"{API_URL}/search",
                json={"query": query, "top_k": top_k, "candidate_pool": candidate_pool},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as error:
            st.error(f"Search failed: {error}")
            st.stop()

    results = data["results"]
    timings = data["timings"]
    total_time = sum(timings.values())

    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Embedding", f"{timings['embedding'] * 1000:.0f} ms")
    col2.metric("Retrieval", f"{timings['retrieval'] * 1000:.0f} ms")
    col3.metric("Reranking", f"{timings.get('reranking', 0) * 1000:.0f} ms")
    col4.metric("Total", f"{total_time * 1000:.0f} ms")

    st.subheader(f"Top {len(results)} Results")

    if not results:
        st.info("No relevant documents found.")
    else:
        for i, result in enumerate(results, start=1):
            with st.container(border=True):
                st.markdown(f"### Result {i}")
                st.write(result["text"])
                c1, c2 = st.columns(2)
                c1.metric("CrossEncoder Score", f"{result['score']:.3f}")
                c2.metric("Cosine Distance", f"{result['distance']:.4f}")
else:
    st.info("Enter a query above to search the document collection.")
