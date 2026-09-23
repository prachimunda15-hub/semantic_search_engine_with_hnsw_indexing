from sentence_transformers import (
    SentenceTransformer
)

from .config import (
    MODEL_NAME,
    BATCH_SIZE
)




model = SentenceTransformer(
    MODEL_NAME,
    device="cuda"
)


def generate_embedding(text):
   

    embedding = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embedding


def generate_embeddings(
    texts,
    batch_size=BATCH_SIZE
):
   

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings


