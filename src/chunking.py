from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from .config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


def create_splitter(tokenizer):
    

    splitter = (
        RecursiveCharacterTextSplitter
        .from_huggingface_tokenizer(
            tokenizer,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
    )

    return splitter


def chunk_documents(rows, tokenizer):
   

    splitter = create_splitter(tokenizer)

    chunks = []

    chunk_source_row = []

    for row_idx, row_text in enumerate(rows):

        row_chunks = splitter.split_text(
            row_text
        )

        chunks.extend(row_chunks)

        chunk_source_row.extend(
            [row_idx] * len(row_chunks)
        )

    return chunks, chunk_source_row
