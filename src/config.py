import os
from dotenv import load_dotenv

load_dotenv()


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

EMBEDDING_DIM = 384



CHUNK_SIZE = 1200

CHUNK_OVERLAP = 200



DATABASE_URL = os.getenv("DATABASE_URL")


if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL is not set. "
        "Add it to your .env file."
    )