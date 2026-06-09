"""
config/settings.py
==================
Central configuration for the Multi-Agent Research Assistant.

All tunable parameters live here — change a value once and it propagates
throughout the entire pipeline. This is the "single source of truth" pattern
used in production ML systems.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables from .env (API keys, secrets)
# ---------------------------------------------------------------------------
load_dotenv()

# Suppress Windows-specific HuggingFace symlink warning.
# Windows requires Developer Mode enabled for symlinks; disabling just
# means the cache uses copy-on-write instead — functionally identical.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ---------------------------------------------------------------------------
# Project Root — used to build absolute paths regardless of where the
# script is invoked from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Google Gemini API
# ---------------------------------------------------------------------------
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# Which Gemini model to use for answer generation.
# - "gemini-1.5-flash" → faster, generous free-tier limits  ← default
# - "gemini-1.5-pro"   → higher reasoning quality, lower RPM on free tier
GEMINI_MODEL: str = "gemini-2.5-flash"

# LLM generation settings
LLM_TEMPERATURE: float = 0.2   # Low temperature = more factual, less creative
LLM_MAX_TOKENS: int = 8192     # Max tokens in the generated answer
                                # Gemini 2.5 Flash supports up to 65k output tokens.
                                # 8192 handles long explanations without runaway cost.

# ---------------------------------------------------------------------------
# Embedding Model (HuggingFace / sentence-transformers)
# ---------------------------------------------------------------------------
# "all-MiniLM-L6-v2" is the industry-standard lightweight embedding model:
#   - 384-dimensional vectors
#   - ~80 MB download (cached after first use)
#   - Runs on CPU, no GPU required
#   - Excellent speed/quality trade-off for document Q&A
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# Where HuggingFace models are cached locally (avoids re-downloading)
HF_CACHE_DIR: Path = PROJECT_ROOT / ".cache" / "huggingface"

# ---------------------------------------------------------------------------
# Text Chunking Parameters
# ---------------------------------------------------------------------------
# chunk_size   → target character count per chunk.
#                Smaller = more precise retrieval but more chunks to search.
#                Larger  = more context per chunk but noisier embeddings.
#                800-1000 chars ≈ 200-250 tokens — a good default for papers.
CHUNK_SIZE: int = 1000

# chunk_overlap → characters shared between consecutive chunks.
#                 Overlap prevents losing context at chunk boundaries.
#                 Rule of thumb: 10-20% of chunk_size.
CHUNK_OVERLAP: int = 200

# ---------------------------------------------------------------------------
# ChromaDB Vector Store
# ---------------------------------------------------------------------------
# Persist directory — ChromaDB saves its index here so re-indexing is not
# required every time the app restarts.
CHROMA_PERSIST_DIR: Path = PROJECT_ROOT / "vectordb"

# Collection name — think of this as a "table" in a traditional database.
# In Phase 2 we will use multiple collections (one per agent / domain).
CHROMA_COLLECTION_NAME: str = "research_docs"

# ---------------------------------------------------------------------------
# Retrieval Parameters
# ---------------------------------------------------------------------------
# Number of chunks to retrieve per query.
# Higher k = more context for the LLM but increases prompt length & cost.
# 4-6 is a good default for document Q&A.
RETRIEVER_TOP_K: int = 5

# ---------------------------------------------------------------------------
# File Storage Paths
# ---------------------------------------------------------------------------
UPLOAD_DIR: Path = PROJECT_ROOT / "data" / "uploads"

# Create directories if they don't exist yet
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
