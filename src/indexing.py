import time

import numpy as np

from .config import EMBEDDING_DIM
from .database import get_connection


def create_embedding_table():

    with get_connection() as conn:

        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS embedding (
                chunk_index BIGSERIAL PRIMARY KEY,
                source_row BIGINT,
                embedding VECTOR({EMBEDDING_DIM}),
                text TEXT
            )
            """
        )

        conn.commit()

        print(
            "Embedding table ready."
        )


def bulk_insert(
    chunks,
    embeddings,
    chunk_source_row=None
):
    """
    Insert embeddings into PostgreSQL
    using PostgreSQL COPY BINARY.
    """

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    
    assert embeddings.shape == (
        len(chunks),
        EMBEDDING_DIM
    ), embeddings.shape


    
    if chunk_source_row is None:

        chunk_source_row = [
            None
        ] * len(chunks)


    assert len(chunk_source_row) == len(chunks)


    with get_connection() as conn:

        
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS embedding (
                chunk_index BIGSERIAL PRIMARY KEY,
                source_row BIGINT,
                embedding VECTOR({EMBEDDING_DIM}),
                text TEXT
            )
            """
        )

        conn.commit()


        
        t0 = time.time()

        with conn.cursor() as cur:

            with cur.copy(
                """
                COPY embedding
                (source_row, text, embedding)
                FROM STDIN
                WITH (FORMAT BINARY)
                """
            ) as copy:

                copy.set_types([
                    "int8",
                    "text",
                    "vector"
                ])

                for source_row, text, embedding in zip(
                    chunk_source_row,
                    chunks,
                    embeddings
                ):

                    copy.write_row(
                        (
                            source_row,
                            text,
                            embedding
                        )
                    )


        t1 = time.time()

        print(
            f"COPY took: "
            f"{t1 - t0:.1f}s"
        )


        
        conn.commit()

        t2 = time.time()

        print(
            f"Commit took: "
            f"{t2 - t1:.1f}s"
        )

        print(
            f"Rows inserted: "
            f"{len(chunks)}"
        )


def create_hnsw_index(
    m=16,
    ef_construction=64
):
    

    with get_connection() as conn:

        print(
            "Creating HNSW index..."
        )

        t0 = time.time()

        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS
            embedding_hnsw
            ON embedding
            USING hnsw (
                embedding vector_cosine_ops
            )
            WITH (
                m = {m},
                ef_construction = {ef_construction}
            )
            """
        )

        conn.commit()

        print(
            f"HNSW index created in "
            f"{time.time() - t0:.1f}s"
        )
