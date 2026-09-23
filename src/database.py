import psycopg

from pgvector.psycopg import register_vector

from .config import DATABASE_URL


def get_connection():
   

    conn = psycopg.connect(
        DATABASE_URL
    )

    register_vector(conn)

    return conn


def create_extension():
   

    with get_connection() as conn:

        conn.execute(
            "CREATE EXTENSION IF NOT EXISTS vector"
        )

        conn.commit()


def test_connection():
    

    with get_connection() as conn:

        result = conn.execute(
           
        ).fetchone()

        print(
            f"Connected as: {result[0]}"
        )

        print(
            f"Database: {result[1]}"
        )