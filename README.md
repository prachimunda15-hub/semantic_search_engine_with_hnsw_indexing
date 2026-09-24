# semantic_search_engine_with_hnsw_indexing

A two-stage semantic search engine over ~120k text passages. Passages are embedded with a Sentence Transformer and stored in PostgreSQL (pgvector). An HNSW index retrieves candidates, and a cross-encoder reranks them. A Streamlit app serves it.


How it works:
OFFLINE
  chunked passages (CSV) -> embed (MiniLM, 384-d) -> Postgres/pgvector -> HNSW index
ONLINE
  query -> embed -> HNSW top-50 -> cross-encoder rerank -> top-5 -> Streamlit

  
  
Stage 1, retrieval. The query is embedded with the same model used for the passages. pgvector returns the 100 nearest passages by cosine distance via the HNSW index. This stage is fast but scores query and passage independently.


Stage 2, reranking. A cross-encoder reads each (query, passage) pair together and produces a relevance score. It is more accurate but too slow to run over the whole corpus, so it only sees the 100 candidates from Stage 1. The top 10 by reranker score are shown.

Stack:
Component	   Choice
Embedding    model	sentence-transformers/all-MiniLM-L6-v2 (384-d)
Reranker	   cross-encoder/ms-marco-MiniLM-L-6-v2
Database	   PostgreSQL + pgvector (hosted on Neon)
Index        HNSW, cosine distance (vector_cosine_ops)
UI	         Streamlit
Language	   Python 3.11 

Data:

Columns used: finalpassage is the text that gets embedded and searched. query is kept for evaluation.


Web Application
The search engine is exposed through a Streamlit application.
The user enters a natural-language query:
How does machine learning work?
The application then:
* Generates the query embedding.
* Searches the HNSW index.
* Retrieves the top 100 candidate passages.
* Applies Cross-Encoder reranking.
* Selects the top 10 passages.
* Displays the results in the Streamlit interface.


License
This project is licensed under the MIT License.


Acknowledgements
* Microsoft Research for the MS MARCO dataset
* Sentence Transformers for the embedding models
* pgvector for PostgreSQL vector similarity search
* PostgreSQL for database infrastructure
* Neon for hosted PostgreSQL
* Streamlit for the application interface
* The open-source Python community















