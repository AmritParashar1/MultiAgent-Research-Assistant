# Multi-Agent Research Assistant

> **Phase 1 — Foundational RAG (Retrieval-Augmented Generation) Pipeline**
>
> A production-quality document Q&A system built as a major Generative AI project.

---

## Architecture Overview

```
User Query
    │
    ▼
[Streamlit UI]  ←──────────────────────────────┐
    │                                           │
    ▼                                           │
[PDF Upload]                              [Answer + Sources]
    │                                           │
    ▼                                           │
[app/loader.py]      PyMuPDF extraction         │
    │                                           │
    ▼                                           │
[app/splitter.py]    Recursive chunking         │
    │                                           │
    ▼                                           │
[app/embeddings.py]  HuggingFace MiniLM-L6-v2  │
    │                                           │
    ▼                                           │
[app/vectorstore.py] ChromaDB (persist)         │
    │                                           │
    ├──► [Retriever] ◄── User Query (embedded) ─┤
    │         │                                 │
    │         ▼                                 │
    │   Top-k Chunks (MMR)                      │
    │         │                                 │
    │         ▼                                 │
    └──► [app/qa_chain.py]  Gemini 1.5 Flash ──►┘
```

---

## Project Structure

```
MultiAgent Research Assistant/
├── app/
│   ├── __init__.py         # Public pipeline API
│   ├── loader.py           # PDF → Documents (PyMuPDF)
│   ├── splitter.py         # Documents → Chunks (RecursiveCharacterTextSplitter)
│   ├── embeddings.py       # HuggingFace embedding model wrapper
│   ├── vectorstore.py      # ChromaDB indexing & retrieval
│   └── qa_chain.py         # LangChain LCEL + Gemini RAG chain
├── ui/
│   └── streamlit_app.py    # Streamlit frontend
├── config/
│   └── settings.py         # Centralised configuration (single source of truth)
├── data/
│   └── uploads/            # Uploaded PDFs (gitignored)
├── vectordb/               # Persisted ChromaDB index (gitignored)
├── tests/
│   └── test_pipeline.py    # Unit tests (pytest)
├── .env.example            # API key template
├── requirements.txt        # Pinned dependencies
└── README.md               # This file
```

---

## Setup & Installation

### Prerequisites
- Python 3.10 or newer
- A [Google Gemini API key](https://aistudio.google.com/app/apikey) (free tier works)

### 1. Clone / open the project

```bash
cd "e:\Projects\MultiAgent Research Assistant"
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note**: The first run will download the HuggingFace embedding model (~80 MB). It is cached in `.cache/huggingface/` for all subsequent runs.

### 4. Configure your API key

```bash
copy .env.example .env          # Windows
# or
cp .env.example .env            # macOS/Linux
```

Open `.env` and set your key:
```
GOOGLE_API_KEY=your_actual_key_here
```

### 5. Run the application

```bash
streamlit run ui/streamlit_app.py
```

The app opens at `http://localhost:8501`.

---

## Usage

1. **Upload PDFs** — Drag and drop one or more PDF files in the sidebar
2. **Index Documents** — Click "⚡ Index Documents" to run the full pipeline
3. **Ask Questions** — Type any question about your documents in the chat input
4. **View Sources** — Expand the "📚 View Sources" panel under each answer

---

## Running Tests

```bash
pytest tests/test_pipeline.py -v
```

Tests cover:
- Document metadata validation
- Chunking correctness (size, inheritance, chunk_id)
- Embedding dimensionality (384 for MiniLM-L6-v2)
- Semantic similarity validation
- LRU cache behaviour

No API key is required to run tests — only local components are tested.

---

## Configuration

All parameters are in `config/settings.py`:

| Parameter | Default | Description |
|---|---|---|
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model for answer generation |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace sentence transformer |
| `CHUNK_SIZE` | `1000` | Target characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `RETRIEVER_TOP_K` | `5` | Number of chunks retrieved per query |
| `LLM_TEMPERATURE` | `0.2` | Lower = more deterministic answers |

---

## Key Concepts

### Why Chunking?
LLMs have finite context windows. Chunking divides documents into focused units so retrieval can surface the **exact relevant passage** rather than entire documents. Overlap (200 chars) ensures no information is lost at chunk boundaries.

### Why Embeddings?
Raw text cannot be compared mathematically. Embeddings convert text into 384-dimensional vectors where semantic similarity = geometric proximity. This enables **meaning-based** search — "cardiac arrest" matches "heart failure" even without shared words.

### How Retrieval Works
1. User query → embedded into vector space
2. ChromaDB performs cosine similarity search (HNSW graph)
3. MMR re-ranking selects top-5 diverse, relevant chunks
4. Chunks → assembled into context → passed to Gemini

### How RAG Reduces Hallucination
The LLM is explicitly instructed to answer **only from the provided context**. If context doesn't contain the answer, it says "I don't know." Every answer is traceable to a specific chunk and page.

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| PDF Extraction | PyMuPDF | Best complex-layout handling |
| Text Splitting | LangChain RecursiveCharacterTextSplitter | Natural boundary awareness |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Fast, CPU-friendly, 384-dim |
| Vector Database | ChromaDB | Embedded, persistent, LangChain-native |
| LLM | Google Gemini 1.5 Flash | Fast, free tier, 1M context window |
| Orchestration | LangChain LCEL | Modern streaming-compatible API |
| Frontend | Streamlit | Rapid prototyping, Python-native |

---

## Roadmap

- **Phase 1** ✅ Foundational RAG Pipeline (this)
- **Phase 2** 🔄 Multi-Agent Architecture (Planner + Researcher + Synthesiser agents)
- **Phase 3** 🔄 Web search integration + cross-document reasoning
- **Phase 4** 🔄 Evaluation framework (RAGAS metrics)

---

## License

MIT License — free for academic and commercial use.
