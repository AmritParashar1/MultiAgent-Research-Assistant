"""
app/__init__.py
===============
Makes 'app' a Python package and exposes the high-level pipeline API.

Other modules can do:
    from app import load_pdfs, chunk_documents, build_vectorstore, get_qa_chain
"""

from app.loader import load_pdfs
from app.splitter import chunk_documents
from app.vectorstore import build_vectorstore, get_retriever
from app.qa_chain import get_qa_chain

__all__ = [
    "load_pdfs",
    "chunk_documents",
    "build_vectorstore",
    "get_retriever",
    "get_qa_chain",
]
