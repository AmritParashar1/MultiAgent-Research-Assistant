"""
app/vectorstore.py
==================
Vector Database Layer

Responsibility:
    - Build (index) a ChromaDB collection from document chunks
    - Provide a LangChain-compatible retriever for semantic search

What is a vector store?
    A vector store is a specialised database that stores embeddings alongside
    their original text. Given a query embedding, it efficiently finds the
    most semantically similar stored embeddings using Approximate Nearest
    Neighbour (ANN) search — ChromaDB uses HNSW (Hierarchical Navigable
    Small World) graphs for this.

Why ChromaDB?
    - Lightweight, embedded (no separate server required for dev)
    - Persistent storage to disk — survives app restarts
    - Native LangChain integration
    - Suitable for collections up to ~1M documents
    - Easy to swap for Pinecone/Weaviate/pgvector in Phase 2

Persistence strategy:
    ChromaDB is configured with a persist_directory so the indexed embeddings
    are saved to disk after each session. The app checks on startup whether an
    index already exists and loads it without re-indexing.
"""

from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from app.embeddings import get_embedding_model
from config.settings import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    RETRIEVER_TOP_K,
)


def build_vectorstore(chunks: list[Document]) -> Chroma:
    """
    Embed all document chunks and store them in ChromaDB.

    This function:
    1. Obtains the cached embedding model
    2. Creates (or resets) a ChromaDB collection
    3. Embeds all chunks in batches and stores them with metadata
    4. Persists the index to disk

    Args:
        chunks: List of Document chunks from splitter.chunk_documents()

    Returns:
        Chroma instance that wraps the persisted collection.

    Notes:
        - If the collection already exists, it is REPLACED. This ensures a
          clean index when new documents are uploaded.
        - In Phase 2 we will add incremental indexing to avoid re-embedding
          documents that haven't changed.
    """
    if not chunks:
        raise ValueError("[VectorStore] No chunks provided — cannot build index.")

    embedding_model = get_embedding_model()

    print(f"[VectorStore] Building index: {len(chunks)} chunks → {CHROMA_COLLECTION_NAME}")
    print(f"[VectorStore] Persist directory: {CHROMA_PERSIST_DIR}")

    # -----------------------------------------------------------------------
    # Delete the existing collection if it exists so uploads are fresh.
    # Use a persistent client so data survives process restarts.
    # -----------------------------------------------------------------------
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))

    # Delete old collection if it exists (idempotent — no error if absent)
    try:
        client.delete_collection(name=CHROMA_COLLECTION_NAME)
        print(f"[VectorStore] Cleared existing collection '{CHROMA_COLLECTION_NAME}'")
    except Exception:
        pass  # Collection didn't exist yet — that's fine

    # -----------------------------------------------------------------------
    # Chroma.from_documents() handles:
    #   1. Calling embedding_model.embed_documents(texts) for all chunks
    #   2. Storing (embedding, text, metadata) tuples in ChromaDB
    #   3. Returning a Chroma wrapper object for retrieval
    # -----------------------------------------------------------------------
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(CHROMA_PERSIST_DIR),
        client=client,
    )

    count = vectorstore._collection.count()
    print(f"[VectorStore] ✓ Indexed {count} vectors into '{CHROMA_COLLECTION_NAME}'")

    return vectorstore


def load_vectorstore() -> Chroma:
    """
    Load an existing ChromaDB collection from disk without re-indexing.

    Used on app startup when a persisted index already exists.

    Returns:
        Chroma instance wrapping the persisted collection.

    Raises:
        RuntimeError: If no persisted index is found (user must upload docs first).
    """
    persist_path = Path(CHROMA_PERSIST_DIR)

    # A ChromaDB SQLite file is the canonical marker that an index exists
    chroma_db_file = persist_path / "chroma.sqlite3"
    if not chroma_db_file.exists():
        raise RuntimeError(
            "[VectorStore] No persisted index found. "
            "Please upload and index documents first."
        )

    embedding_model = get_embedding_model()
    client = chromadb.PersistentClient(path=str(persist_path))

    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(persist_path),
        client=client,
    )

    count = vectorstore._collection.count()
    print(f"[VectorStore] ✓ Loaded existing index: {count} vectors in '{CHROMA_COLLECTION_NAME}'")

    return vectorstore


def get_retriever(vectorstore: Chroma) -> VectorStoreRetriever:
    """
    Create a LangChain retriever from an indexed Chroma collection.

    The retriever is the bridge between the vector store and the QA chain.
    When called with a query string, it:
        1. Embeds the query using the same embedding model
        2. Performs cosine similarity search in ChromaDB
        3. Returns the top-k most similar Document chunks

    Args:
        vectorstore: A Chroma instance (from build_vectorstore or load_vectorstore)

    Returns:
        VectorStoreRetriever configured for top-k similarity search.

    Retrieval types supported by Chroma:
        "similarity"         — standard cosine similarity (default)
        "mmr"                — Maximum Marginal Relevance: balances relevance
                               AND diversity, reduces redundant retrieved chunks
        "similarity_score_threshold" — only return chunks above a score cutoff
    """
    retriever = vectorstore.as_retriever(
        search_type="mmr",   # MMR = better diversity among retrieved chunks
        search_kwargs={
            "k": RETRIEVER_TOP_K,
            "fetch_k": RETRIEVER_TOP_K * 3,  # MMR: fetch 3x candidates, re-rank to k
            "lambda_mult": 0.7,              # MMR diversity weight (0=max diversity, 1=max relevance)
        },
    )

    print(f"[VectorStore] Retriever ready (MMR, k={RETRIEVER_TOP_K})")
    return retriever
