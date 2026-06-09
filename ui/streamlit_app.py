"""
ui/streamlit_app.py
===================
Streamlit Frontend for the Multi-Agent Research Assistant

Architecture:
    This file is the entry-point for the web UI. It orchestrates the pipeline
    modules (loader → splitter → vectorstore → qa_chain) and manages all UI state.

Streamlit session state strategy:
    Streamlit re-runs the entire script on every user interaction. Session
    state (st.session_state) acts as persistent memory across re-runs within
    the same browser session. We store:
        - vectorstore    : the indexed Chroma collection (avoids re-indexing)
        - qa_chain       : the compiled RAG chain (avoids re-compilation)
        - chat_history   : list of (question, answer, sources) tuples
        - indexed_files  : names of currently indexed PDFs
"""

import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — ensure project root is on sys.path so 'app' and 'config'
# can be imported regardless of where streamlit is launched from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.loader import load_pdfs
from app.splitter import chunk_documents
from app.vectorstore import build_vectorstore, load_vectorstore, get_retriever
from app.qa_chain import get_qa_chain
from config.settings import CHROMA_PERSIST_DIR, UPLOAD_DIR


# ===========================================================================
# PAGE CONFIGURATION
# ===========================================================================
st.set_page_config(
    page_title="Research Assistant — RAG Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": "Multi-Agent Research Assistant | Phase 1 — RAG Pipeline",
    },
)

# ===========================================================================
# CUSTOM CSS
# ===========================================================================
st.markdown("""
<style>
    /* ---- Google Font ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ---- Root variables ---- */
    :root {
        --bg-primary: #0d1117;
        --bg-secondary: #161b22;
        --bg-card: #1c2130;
        --accent-blue: #4f8ef7;
        --accent-purple: #7c5cbf;
        --accent-green: #3fb950;
        --accent-orange: #f78166;
        --text-primary: #e6edf3;
        --text-secondary: #8b949e;
        --border-color: #30363d;
        --gradient: linear-gradient(135deg, #4f8ef7 0%, #7c5cbf 100%);
    }

    /* ---- Global ---- */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }

    .main { background-color: var(--bg-primary); }
    .stApp { background-color: var(--bg-primary); }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--text-primary);
    }

    /* ---- Header gradient title ---- */
    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: var(--gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        color: var(--text-secondary);
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* ---- Chat bubbles ---- */
    .chat-user {
        background: linear-gradient(135deg, #1e3a5f 0%, #1a2d4a 100%);
        border: 1px solid #2d5a8e;
        border-radius: 12px 12px 4px 12px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        color: var(--text-primary);
    }
    .chat-assistant {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px 12px 12px 4px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        color: var(--text-primary);
        line-height: 1.7;
    }
    .chat-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
    }
    .label-user { color: #4f8ef7; }
    .label-assistant { color: #7c5cbf; }

    /* ---- Source card ---- */
    .source-card {
        background: #111827;
        border: 1px solid #2d3748;
        border-left: 3px solid var(--accent-purple);
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.85rem;
        line-height: 1.6;
    }
    .source-meta {
        color: var(--accent-purple);
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .source-text {
        color: #9ca3af;
        font-size: 0.82rem;
    }

    /* ---- Status badges ---- */
    .status-ready {
        display: inline-block;
        background: #132a1a;
        border: 1px solid var(--accent-green);
        color: var(--accent-green);
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-empty {
        display: inline-block;
        background: #2a1a13;
        border: 1px solid var(--accent-orange);
        color: var(--accent-orange);
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* ---- Metric cards ---- */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: var(--gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-label {
        color: var(--text-secondary);
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ---- Pipeline flow diagram ---- */
    .pipeline-step {
        display: inline-block;
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 0.3rem 0.7rem;
        font-size: 0.78rem;
        color: var(--text-secondary);
        margin: 0.15rem;
    }
    .pipeline-step.active {
        border-color: var(--accent-blue);
        color: var(--accent-blue);
    }

    /* ---- Dividers ---- */
    hr { border-color: var(--border-color); }

    /* ---- Input box ---- */
    .stTextInput input, .stTextArea textarea {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 2px rgba(79, 142, 247, 0.2) !important;
    }

    /* ---- Buttons ---- */
    .stButton > button {
        background: var(--gradient);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: opacity 0.2s ease, transform 0.1s ease;
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ---- File uploader ---- */
    [data-testid="stFileUploader"] {
        border: 2px dashed var(--border-color) !important;
        border-radius: 10px !important;
        background: var(--bg-card) !important;
        padding: 1rem !important;
        transition: border-color 0.2s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent-blue) !important;
    }

    /* ---- Expander ---- */
    [data-testid="stExpander"] {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }

    /* ---- Scrollbar ---- */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent-blue); }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# SESSION STATE INITIALISATION
# ===========================================================================
def init_session_state():
    """Initialise all session state keys with safe defaults on first run."""
    defaults = {
        "vectorstore": None,
        "qa_chain": None,
        "chat_history": [],     # List of dicts: {question, answer, sources}
        "indexed_files": [],    # Names of currently indexed PDFs
        "index_stats": {},      # {chunks: int, files: int}
        "is_indexing": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================
def save_uploaded_files(uploaded_files) -> list[Path]:
    """Save Streamlit UploadedFile objects to disk and return their paths."""
    saved_paths = []
    for uploaded_file in uploaded_files:
        dest = UPLOAD_DIR / uploaded_file.name
        with open(dest, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(dest)
    return saved_paths


def index_documents(uploaded_files, status_placeholder) -> tuple[int, int]:
    """
    Run the full indexing pipeline: Load → Chunk → Embed → Store.

    Args:
        uploaded_files: Streamlit UploadedFile list from the file uploader.
        status_placeholder: A st.empty() placeholder for progress messages.

    Returns:
        (num_chunks, num_pages) tuple for display in metrics.

    Note:
        All st.spinner / st.status calls are deliberately removed from this
        function. Streamlit's RerunException (raised by st.rerun()) must never
        be caught inside a try/except block — keeping UI concerns in the caller
        avoids that bug entirely.
    """
    # 1. Save uploaded files to disk
    status_placeholder.info("📄 Saving uploaded files...")
    file_paths = save_uploaded_files(uploaded_files)

    # 2. Load PDFs → Documents
    status_placeholder.info("📄 Extracting text from PDFs...")
    pages = load_pdfs(file_paths)

    if not pages:
        raise ValueError(
            "No text could be extracted from the uploaded PDFs. "
            "Make sure they are text-based (not scanned images)."
        )

    # 3. Chunk Documents
    status_placeholder.info(f"✂️ Splitting {len(pages)} pages into semantic chunks...")
    chunks = chunk_documents(pages)

    # 4. Build vector store (this is the slow step — embedding all chunks)
    status_placeholder.info(
        f"🧠 Generating embeddings for {len(chunks)} chunks & building vector index..."
    )
    vectorstore = build_vectorstore(chunks)

    # 5. Create retriever and QA chain
    status_placeholder.info("🔗 Wiring up retriever and QA chain...")
    retriever = get_retriever(vectorstore)
    qa_chain = get_qa_chain(retriever)

    # 6. Persist everything to session state
    st.session_state.vectorstore = vectorstore
    st.session_state.qa_chain = qa_chain
    st.session_state.indexed_files = [f.name for f in uploaded_files]
    st.session_state.index_stats = {
        "chunks": len(chunks),
        "pages": len(pages),
        "files": len(uploaded_files),
    }
    st.session_state.chat_history = []  # Clear history on re-index

    return len(chunks), len(pages)


# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    # --- Logo / Title ---
    st.markdown("""
    <div style='padding: 0.5rem 0 1rem 0;'>
        <div style='font-size: 1.5rem; font-weight: 700; background: linear-gradient(135deg, #4f8ef7, #7c5cbf); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;'>
            🔬 Research Assistant
        </div>
        <div style='color: #8b949e; font-size: 0.82rem; margin-top: 0.2rem;'>
            Phase 1 — RAG Pipeline
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # --- Status indicator ---
    is_ready = st.session_state.qa_chain is not None
    if is_ready:
        st.markdown(f'<span class="status-ready">● Index Ready</span>', unsafe_allow_html=True)
        st.caption(f"📁 {', '.join(st.session_state.indexed_files)}")
    else:
        st.markdown('<span class="status-empty">○ No Index Loaded</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- File Uploader ---
    st.markdown("#### 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        label="Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more research papers, reports, or any PDF documents.",
        key="pdf_uploader",
    )

    # --- Index Button ---
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button(
        "⚡ Index Documents",
        use_container_width=True,
        disabled=not uploaded_files,
        help="Extract text, generate embeddings, and build the vector index.",
    ):
        # ---------------------------------------------------------------
        # BUG FIX: st.rerun() raises a RerunException internally.
        # If st.rerun() is called inside a try/except Exception block, the
        # exception is caught and the rerun silently fails.
        # Solution: run the pipeline in try/except, store the result/error
        # in local variables, then call st.rerun() OUTSIDE the try block.
        # ---------------------------------------------------------------
        _status = st.empty()   # Single placeholder for all progress messages
        _error = None
        _result = None

        try:
            _result = index_documents(uploaded_files, _status)
        except Exception as e:
            _error = e

        # Handle outcome — st.rerun() is now safely outside try/except
        if _error:
            _status.error(f"❌ Indexing failed: {_error}")
            import traceback
            st.code(traceback.format_exc(), language="python")
        elif _result:
            n_chunks, n_pages = _result
            _status.success(f"✅ Indexed {n_chunks} chunks from {n_pages} pages — Ready!")
            st.rerun()   # Safe: outside try/except, RerunException propagates correctly

    # --- Try loading existing index ---
    if not is_ready:
        chroma_db = CHROMA_PERSIST_DIR / "chroma.sqlite3"
        if chroma_db.exists():
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("♻️ Load Existing Index", use_container_width=True,
                          help="Load the previously built vector index from disk."):
                try:
                    with st.spinner("Loading index from disk..."):
                        vs = load_vectorstore()
                        retriever = get_retriever(vs)
                        st.session_state.vectorstore = vs
                        st.session_state.qa_chain = get_qa_chain(retriever)
                        count = vs._collection.count()
                        st.session_state.index_stats = {"chunks": count, "pages": "?", "files": "?"}
                    st.success("Index loaded!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not load index: {e}")

    st.divider()

    # --- Index Statistics ---
    if is_ready and st.session_state.index_stats:
        stats = st.session_state.index_stats
        st.markdown("#### 📊 Index Statistics")
        cols = st.columns(2)
        with cols[0]:
            st.metric("Chunks", stats.get("chunks", "—"))
        with cols[1]:
            st.metric("Pages", stats.get("pages", "—"))
        st.metric("Files", stats.get("files", "—"))

    st.divider()

    # --- Pipeline Info ---
    st.markdown("#### ⚙️ Pipeline")
    st.markdown("""
    <div style='font-size: 0.8rem; color: #8b949e; line-height: 1.8;'>
    📄 <b>Loader</b>: PyMuPDF<br>
    ✂️ <b>Splitter</b>: Recursive (1000c/200o)<br>
    🧠 <b>Embeddings</b>: MiniLM-L6-v2<br>
    🗄️ <b>Vector DB</b>: ChromaDB (MMR)<br>
    💬 <b>LLM</b>: Gemini 1.5 Flash
    </div>
    """, unsafe_allow_html=True)

    # --- Clear chat ---
    if st.session_state.chat_history:
        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ===========================================================================
# MAIN PANEL
# ===========================================================================
# --- Header ---
st.markdown("""
<div class="app-title">Multi-Agent Research Assistant</div>
<div class="app-subtitle">Phase 1 — Retrieval-Augmented Generation (RAG) Pipeline</div>
""", unsafe_allow_html=True)

# --- Pipeline Diagram ---
st.markdown("""
<div style='margin-bottom: 1.5rem;'>
    <span class="pipeline-step active">📄 PDF Upload</span>
    <span style='color: #4f8ef7;'>→</span>
    <span class="pipeline-step active">✂️ Chunking</span>
    <span style='color: #4f8ef7;'>→</span>
    <span class="pipeline-step active">🧠 Embedding</span>
    <span style='color: #4f8ef7;'>→</span>
    <span class="pipeline-step active">🗄️ ChromaDB</span>
    <span style='color: #4f8ef7;'>→</span>
    <span class="pipeline-step active">🔍 Retrieval</span>
    <span style='color: #4f8ef7;'>→</span>
    <span class="pipeline-step active">💬 Gemini</span>
    <span style='color: #4f8ef7;'>→</span>
    <span class="pipeline-step active">📋 Answer + Sources</span>
</div>
""", unsafe_allow_html=True)

st.divider()

# ===========================================================================
# CHAT INTERFACE
# ===========================================================================

# --- Display chat history ---
if not st.session_state.chat_history:
    if st.session_state.qa_chain:
        # Index is ready — show prompt
        st.markdown("""
        <div style='text-align: center; padding: 3rem 0; color: #8b949e;'>
            <div style='font-size: 2.5rem; margin-bottom: 1rem;'>💬</div>
            <div style='font-size: 1.1rem; font-weight: 500; color: #e6edf3;'>
                Your documents are indexed and ready.
            </div>
            <div style='font-size: 0.9rem; margin-top: 0.5rem;'>
                Ask any question about your uploaded research papers below.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # No index — show onboarding
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0; color: #8b949e;'>
            <div style='font-size: 2.5rem; margin-bottom: 1rem;'>🔬</div>
            <div style='font-size: 1.1rem; font-weight: 500; color: #e6edf3;'>
                Welcome to the Research Assistant
            </div>
            <div style='font-size: 0.9rem; margin-top: 0.5rem;'>
                Upload your PDF documents in the sidebar and click <b>Index Documents</b> to get started.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Feature cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div style='font-size: 1.5rem;'>📄</div>
                <div style='font-weight: 600; margin: 0.3rem 0;'>Upload PDFs</div>
                <div style='color: #8b949e; font-size: 0.82rem;'>Research papers, reports, textbooks</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div style='font-size: 1.5rem;'>🔍</div>
                <div style='font-weight: 600; margin: 0.3rem 0;'>Semantic Search</div>
                <div style='color: #8b949e; font-size: 0.82rem;'>Finds meaning, not just keywords</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div style='font-size: 1.5rem;'>📋</div>
                <div style='font-weight: 600; margin: 0.3rem 0;'>Source Attribution</div>
                <div style='color: #8b949e; font-size: 0.82rem;'>Every answer is traceable</div>
            </div>
            """, unsafe_allow_html=True)
else:
    # Render conversation history
    for turn in st.session_state.chat_history:

        # --- User bubble ---
        # Using st.chat_message instead of raw HTML f-string interpolation.
        # Raw interpolation breaks silently when the answer contains HTML
        # special characters like <, >, ^ (common in architecture/math answers).
        with st.chat_message("user", avatar="👤"):
            st.markdown(turn['question'])

        # --- Assistant bubble ---
        with st.chat_message("assistant", avatar="🔬"):
            # st.markdown() renders Gemini's output as proper markdown:
            # bold, bullets, numbered lists, code blocks, math — all work correctly.
            # HTML special characters are automatically escaped.
            st.markdown(turn['answer'])

        # --- Source attribution ---
        if turn.get("sources"):
            with st.expander(f"📚 View Sources ({len(turn['sources'])} chunks retrieved)"):
                for i, doc in enumerate(turn["sources"], start=1):
                    source = doc.metadata.get("source", "Unknown")
                    page   = doc.metadata.get("page", "?")
                    text_preview = doc.page_content[:400] + (
                        "..." if len(doc.page_content) > 400 else ""
                    )
                    st.markdown(f"""
<div class="source-card">
    <div class="source-meta">Source {i} · {source} · Page {page}</div>
    <div class="source-text">{text_preview}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

# --- Query input ---
st.divider()

with st.form(key="query_form", clear_on_submit=True):
    col_input, col_btn = st.columns([6, 1])
    with col_input:
        user_query = st.text_input(
            label="Ask a question",
            placeholder="e.g. What methodology did the authors use? What are the key findings?",
            label_visibility="collapsed",
            key="query_input",
        )
    with col_btn:
        submit = st.form_submit_button(
            "Ask →",
            use_container_width=True,
            disabled=not st.session_state.qa_chain,
        )

if not st.session_state.qa_chain:
    st.caption("⬆ Upload and index documents to enable the Q&A interface.")


# ===========================================================================
# QUERY PROCESSING
# ===========================================================================
if submit and user_query and st.session_state.qa_chain:
    with st.spinner("🔍 Retrieving relevant context and generating answer..."):
        try:
            result = st.session_state.qa_chain(user_query)

            # Append to chat history
            st.session_state.chat_history.append({
                "question": user_query,
                "answer": result["answer"],
                "sources": result["source_documents"],
            })

            st.rerun()

        except Exception as e:
            st.error(f"Error generating answer: {e}")
            if "GOOGLE_API_KEY" in str(e) or "API_KEY" in str(e).upper():
                st.info("💡 Check that your GOOGLE_API_KEY is set correctly in the .env file.")
