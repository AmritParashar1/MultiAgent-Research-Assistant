"""
tests/test_pipeline.py
======================
Unit tests for each layer of the RAG pipeline.

Run with:
    pytest tests/test_pipeline.py -v

These tests:
    1. Validate that each module produces the correct output types and shapes
    2. Use a minimal synthetic Document so no actual PDF files are needed
    3. Are fast (no LLM API calls — only local components are tested)
    4. Serve as regression tests when refactoring pipeline code
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path when running tests from any directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from langchain_core.documents import Document

from app.splitter import chunk_documents
from app.embeddings import get_embedding_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_documents() -> list[Document]:
    """
    Minimal synthetic documents that mimic loader.load_pdfs() output.
    Using synthetic data means tests run without any PDF files on disk.
    """
    return [
        Document(
            page_content=(
                "Transformer architecture introduced self-attention mechanisms "
                "which revolutionised natural language processing. The attention "
                "mechanism computes weighted combinations of all positions in a "
                "sequence simultaneously, unlike recurrent networks that process "
                "tokens sequentially.\n\n"
                "The scaled dot-product attention formula is: "
                "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V\n\n"
                "Multi-head attention runs several attention operations in parallel "
                "and concatenates their outputs, allowing the model to attend to "
                "information from different representation subspaces."
            ),
            metadata={"source": "attention_is_all_you_need.pdf", "page": 1, "total_pages": 15},
        ),
        Document(
            page_content=(
                "Retrieval-Augmented Generation (RAG) combines a parametric model "
                "(the LLM) with a non-parametric memory (a dense vector retriever). "
                "Given an input query, RAG first retrieves a set of relevant documents "
                "from a large corpus using maximum inner product search (MIPS).\n\n"
                "The retrieved documents are then concatenated with the original query "
                "and fed to a seq2seq model to generate the final answer. This approach "
                "significantly reduces hallucination because the model's output is "
                "grounded in retrieved evidence rather than purely in parametric memory."
            ),
            metadata={"source": "rag_paper.pdf", "page": 3, "total_pages": 12},
        ),
    ]


@pytest.fixture(scope="module")
def sample_chunks(sample_documents) -> list[Document]:
    """Chunks derived from sample_documents for use in downstream tests."""
    return chunk_documents(sample_documents)


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------

class TestLoader:
    def test_document_has_required_metadata(self, sample_documents):
        """Each document must have source and page metadata."""
        for doc in sample_documents:
            assert "source" in doc.metadata, "Missing 'source' in metadata"
            assert "page" in doc.metadata, "Missing 'page' in metadata"

    def test_document_has_non_empty_content(self, sample_documents):
        """Page content must not be empty."""
        for doc in sample_documents:
            assert doc.page_content.strip(), "Document has empty page_content"


# ---------------------------------------------------------------------------
# Splitter tests
# ---------------------------------------------------------------------------

class TestSplitter:
    def test_chunking_produces_more_units_than_pages(self, sample_documents, sample_chunks):
        """
        Chunking should produce at least as many units as the input documents.
        For multi-paragraph docs, we expect more chunks than pages.
        """
        assert len(sample_chunks) >= len(sample_documents), (
            f"Expected >= {len(sample_documents)} chunks, got {len(sample_chunks)}"
        )

    def test_chunks_inherit_source_metadata(self, sample_chunks):
        """All chunks must retain the 'source' metadata from their parent document."""
        for chunk in sample_chunks:
            assert "source" in chunk.metadata, f"Chunk missing 'source': {chunk.metadata}"

    def test_chunks_have_chunk_id(self, sample_chunks):
        """Chunker must add a 'chunk_id' string to each chunk's metadata."""
        for chunk in sample_chunks:
            assert "chunk_id" in chunk.metadata, f"Chunk missing 'chunk_id': {chunk.metadata}"
            assert isinstance(chunk.metadata["chunk_id"], str)

    def test_chunk_size_within_bounds(self, sample_chunks):
        """
        No chunk should exceed chunk_size + chunk_overlap.
        The splitter may slightly exceed chunk_size at natural boundaries,
        so we allow a generous buffer.
        """
        from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
        max_allowed = CHUNK_SIZE + CHUNK_OVERLAP + 100  # generous buffer

        for chunk in sample_chunks:
            assert len(chunk.page_content) <= max_allowed, (
                f"Chunk too large: {len(chunk.page_content)} chars "
                f"(max allowed: {max_allowed})"
            )

    def test_empty_input_returns_empty_list(self):
        """chunk_documents([]) should return [] without errors."""
        result = chunk_documents([])
        assert result == []

    def test_chunk_content_is_non_empty(self, sample_chunks):
        """No chunk should have empty page_content."""
        for chunk in sample_chunks:
            assert chunk.page_content.strip(), "Chunk has empty page_content"


# ---------------------------------------------------------------------------
# Embeddings tests
# ---------------------------------------------------------------------------

class TestEmbeddings:
    def test_model_loads_successfully(self):
        """Embedding model should load without exceptions."""
        model = get_embedding_model()
        assert model is not None

    def test_embed_query_returns_vector(self):
        """embed_query should return a list of floats."""
        model = get_embedding_model()
        vector = model.embed_query("What is attention in transformers?")
        assert isinstance(vector, list), "Expected a list of floats"
        assert len(vector) > 0, "Embedding vector should not be empty"
        assert all(isinstance(v, float) for v in vector), "All elements should be floats"

    def test_embedding_dimension(self):
        """MiniLM-L6-v2 produces 384-dimensional vectors."""
        model = get_embedding_model()
        vector = model.embed_query("test")
        assert len(vector) == 384, (
            f"Expected 384-dim vector, got {len(vector)}-dim. "
            "If you changed the embedding model, update this test."
        )

    def test_similar_texts_have_higher_similarity(self):
        """
        Two semantically similar texts should produce more similar embeddings
        than two dissimilar texts. This validates that the model is working
        correctly for semantic search.
        """
        import numpy as np

        model = get_embedding_model()

        v_attention = model.embed_query("The attention mechanism in transformers")
        v_similar   = model.embed_query("Self-attention in neural networks")
        v_different = model.embed_query("The French Revolution began in 1789")

        # Cosine similarity (vectors are L2-normalised so dot product = cosine sim)
        sim_similar   = np.dot(v_attention, v_similar)
        sim_different = np.dot(v_attention, v_different)

        assert sim_similar > sim_different, (
            f"Expected similar texts to have higher cosine similarity "
            f"({sim_similar:.3f} vs {sim_different:.3f})"
        )

    def test_lru_cache_returns_same_instance(self):
        """get_embedding_model() should return the same object (cached)."""
        model_a = get_embedding_model()
        model_b = get_embedding_model()
        assert model_a is model_b, "Expected the same cached model instance"
