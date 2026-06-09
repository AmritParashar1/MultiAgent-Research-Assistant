"""
app/loader.py
=============
PDF Ingestion & Text Extraction Layer

Responsibility:
    Load one or more PDF files from disk and convert them into a list of
    LangChain `Document` objects — the standard unit of text + metadata
    throughout the entire pipeline.

Why PyMuPDF (fitz)?
    - Handles complex layouts: multi-column papers, tables, academic PDFs
    - Preserves reading order better than pypdf / pdfminer
    - Significantly faster on large documents
    - Correctly extracts text that pypdf silently drops

Output schema per Document:
    page_content : str   — raw text of one PDF page
    metadata     : dict  — {"source": filename, "page": page_number (1-indexed)}
"""

from pathlib import Path
from typing import Union
import fitz  # PyMuPDF
from langchain_core.documents import Document


def load_pdf(file_path: Union[str, Path]) -> list[Document]:
    """
    Extract text from a single PDF file, one Document per page.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        List of Document objects, one per page with source metadata.

    Raises:
        FileNotFoundError: If the PDF does not exist at the given path.
        ValueError: If the file is not a PDF or is password-protected.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {file_path.suffix}")

    documents: list[Document] = []

    # Open with PyMuPDF — this context manager ensures the file is closed
    # even if an exception occurs mid-extraction.
    with fitz.open(str(file_path)) as pdf:
        if pdf.is_encrypted:
            raise ValueError(f"PDF is password-protected: {file_path.name}")

        for page_index in range(len(pdf)):
            page = pdf[page_index]

            # get_text("text") extracts plain text preserving reading order.
            # Other options: "html", "dict", "rawdict" for structured extraction.
            raw_text: str = page.get_text("text")

            # Skip pages that are purely images / scanned with no text layer
            cleaned_text = raw_text.strip()
            if not cleaned_text:
                continue

            documents.append(
                Document(
                    page_content=cleaned_text,
                    metadata={
                        "source": file_path.name,
                        "page": page_index + 1,         # Human-readable (1-indexed)
                        "total_pages": len(pdf),
                        "file_path": str(file_path),
                    },
                )
            )

    return documents


def load_pdfs(file_paths: list[Union[str, Path]]) -> list[Document]:
    """
    Load and extract text from multiple PDF files.

    This is the primary entry-point used by the pipeline. It batches multiple
    uploads into a single flat list of Documents ready for chunking.

    Args:
        file_paths: List of paths to PDF files.

    Returns:
        Flat list of Documents from all files, in order of input.

    Example:
        >>> docs = load_pdfs(["paper1.pdf", "paper2.pdf"])
        >>> print(f"Loaded {len(docs)} pages across {len(file_paths)} files")
    """
    all_documents: list[Document] = []

    for path in file_paths:
        try:
            docs = load_pdf(path)
            all_documents.extend(docs)
            print(f"[Loader] ✓ Loaded '{Path(path).name}' — {len(docs)} pages extracted")
        except (FileNotFoundError, ValueError) as exc:
            # Log the error but continue processing other files
            print(f"[Loader] ✗ Skipped '{Path(path).name}': {exc}")

    print(f"[Loader] Total pages loaded: {len(all_documents)}")
    return all_documents
