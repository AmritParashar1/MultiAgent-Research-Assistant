"""
app/embeddings.py
=================
Embedding Model Layer

Responsibility:
    Initialise and return the HuggingFace embedding model used to convert
    text (both document chunks and user queries) into dense numerical vectors.

What is an embedding?
    An embedding is a fixed-length numerical vector (e.g., 384 floats for
    MiniLM-L6-v2) that represents the *meaning* of a piece of text.
    Texts with similar meanings produce vectors that are geometrically close
    (high cosine similarity). This enables semantic search — finding
    conceptually related passages even when they share no keywords.

Why sentence-transformers/all-MiniLM-L6-v2?
    - Trained specifically for semantic similarity tasks
    - 384-dimensional output — small enough for fast ANN search
    - ~80 MB model, runs on CPU — no GPU required
    - State-of-the-art on MTEB benchmark for its size class
    - Apache 2.0 license — commercially usable

Important: The SAME embedding model must be used at both indexing time
    (when chunks are embedded and stored) and query time (when the user's
    question is embedded for similarity search). Mixing models will produce
    meaningless results.
"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import EMBEDDING_MODEL, HF_CACHE_DIR


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Return a cached instance of the HuggingFace embedding model.

    Using @lru_cache ensures the model is loaded from disk ONLY ONCE per
    process, even if this function is called many times (e.g., on every
    Streamlit re-render). This avoids expensive repeated model loading.

    Returns:
        HuggingFaceEmbeddings — LangChain-compatible embedding model that
        can be passed directly to ChromaDB's from_documents() method.

    Usage:
        >>> embedder = get_embedding_model()
        >>> vector = embedder.embed_query("What is attention in transformers?")
        >>> print(len(vector))   # → 384
    """
    print(f"[Embeddings] Loading model: {EMBEDDING_MODEL}")
    print(f"[Embeddings] Cache directory: {HF_CACHE_DIR}")

    model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        cache_folder=str(HF_CACHE_DIR),
        model_kwargs={
            "device": "cpu",        # Set to "cuda" if you have a GPU
        },
        encode_kwargs={
            "normalize_embeddings": True,  # L2 normalise → cosine sim = dot product
                                           # Required for ChromaDB's default metric
            "batch_size": 32,              # Process 32 chunks at once for efficiency
        },
    )

    print(f"[Embeddings] ✓ Model ready")
    return model
