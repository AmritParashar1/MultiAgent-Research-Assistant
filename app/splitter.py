"""
app/splitter.py
===============
Text Chunking Layer

Responsibility:
    Take raw page-level Documents (from loader.py) and split them into
    smaller, semantically focused chunks suitable for embedding.

Why chunk at all?
    1. Embedding quality degrades with very long texts — the model "averages"
       over too many concepts, losing precision.
    2. Retrieval must return focused content — a 10-page chapter is useless as
       a single retrieval unit; the 2 relevant paragraphs within it are perfect.
    3. LLMs have context window limits — retrieving 5 focused chunks is much
       more efficient than retrieving 5 full pages.

Why RecursiveCharacterTextSplitter?
    It tries to split on natural boundaries in order:
        1. Paragraphs  (\n\n)
        2. Lines       (\n)
        3. Sentences   (". ")
        4. Words       (" ")
        5. Characters  (fallback)
    This preserves semantic coherence far better than naive fixed-size splitting.

Why overlap?
    When a sentence spans a chunk boundary, both adjacent chunks contain it.
    This prevents retrieval from missing a critical fact because it fell
    exactly on the cut point.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split a list of page-level Documents into smaller overlapping chunks.

    Each output chunk inherits the metadata of its parent page (source, page
    number) so source attribution works correctly downstream.

    Args:
        documents: List of Documents produced by loader.load_pdfs()

    Returns:
        List of chunk-level Documents, ready for embedding and indexing.

    Example:
        >>> pages = load_pdfs(["paper.pdf"])
        >>> chunks = chunk_documents(pages)
        >>> print(f"{len(pages)} pages → {len(chunks)} chunks")
    """
    if not documents:
        return []

    # -----------------------------------------------------------------------
    # RecursiveCharacterTextSplitter configuration
    # -----------------------------------------------------------------------
    # separators: The splitter tries these in order, falling back to the next
    #             if the chunk is still too large.
    # chunk_size: Target size in CHARACTERS (not tokens). Using characters is
    #             model-agnostic; ~1000 chars ≈ 200-250 tokens with MiniLM.
    # chunk_overlap: Characters to repeat between consecutive chunks.
    # add_start_index: Adds the character offset within the original page text
    #                  to metadata — useful for debugging and future features.
    # -----------------------------------------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,   # Adds metadata["start_index"] to each chunk
    )

    chunks: list[Document] = splitter.split_documents(documents)

    # -----------------------------------------------------------------------
    # Add a human-readable chunk ID to each chunk's metadata.
    # Format: "filename.pdf | page 3 | chunk 2"
    # This is used in the UI for source attribution and in Phase 2 for
    # agent memory / deduplication.
    # -----------------------------------------------------------------------
    # Group chunks by their source so chunk numbering resets per document
    chunk_counters: dict[str, int] = {}

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "?")
        key = f"{source}|p{page}"

        chunk_counters[key] = chunk_counters.get(key, 0) + 1
        chunk_num = chunk_counters[key]

        chunk.metadata["chunk_id"] = f"{source} | page {page} | chunk {chunk_num}"

    print(f"[Splitter] {len(documents)} pages → {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    return chunks
