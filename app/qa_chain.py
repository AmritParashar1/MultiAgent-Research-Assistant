"""
app/qa_chain.py
===============
Question-Answering (RAG) Chain Layer

Responsibility:
    Compose the retriever and the Gemini LLM into a complete RAG pipeline
    that takes a user question and returns a grounded, source-attributed answer.

The RAG flow at query time:
    ┌────────────────────────────────────────────────────────────────┐
    │  User Query                                                    │
    │      │                                                         │
    │      ▼                                                         │
    │  [Embed Query] ──► [ChromaDB Similarity Search] ──► [Top-k Chunks]
    │                                                         │      │
    │                                                         │      │
    │      ┌──────────────────────────────────────────────────┘      │
    │      │                                                         │
    │      ▼                                                         │
    │  [Build Prompt] = System prompt + Retrieved context + Query    │
    │      │                                                         │
    │      ▼                                                         │
    │  [Gemini LLM] → Answer (grounded in retrieved context)        │
    │      │                                                         │
    │      ▼                                                         │
    │  Return: {"answer": str, "source_documents": [Document]}      │
    └────────────────────────────────────────────────────────────────┘

How RAG reduces hallucination:
    The prompt explicitly instructs the LLM to answer ONLY using the
    provided context. If the answer is not in the context, it is instructed
    to say "I don't know" rather than fabricating an answer. This is the
    key mechanism that makes RAG trustworthy for document Q&A.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document

from config.settings import (
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)


# ---------------------------------------------------------------------------
# Prompt Engineering
# ---------------------------------------------------------------------------
# This is the single most impactful lever for RAG quality.
# Key design decisions:
#   1. "Only use the following context" — prevents the LLM from using
#      parametric memory (which may be stale or incorrect)
#   2. "If you don't know, say so" — prevents hallucination when context
#      doesn't contain the answer
#   3. Structured format — asks for a concise answer then supporting detail
#      makes responses more scannable
# ---------------------------------------------------------------------------
RAG_SYSTEM_PROMPT = """You are a precise research assistant specialising in \
academic and technical document analysis.

Your task is to answer the user's question using ONLY the information \
provided in the context below. The context consists of excerpts retrieved \
from one or more research documents.

Guidelines:
- Base your answer entirely on the provided context. Do not use outside knowledge.
- If the context does not contain enough information to answer the question, \
say "I don't have enough information in the provided documents to answer this."
- Be concise but thorough. Use bullet points or numbered lists where appropriate.
- When referencing specific information, note which source it comes from \
(e.g., "According to [filename], page X...").
- Preserve technical terminology exactly as it appears in the source.

Context:
{context}
"""

RAG_HUMAN_PROMPT = "Question: {question}"


def _format_docs(docs: list[Document]) -> str:
    """
    Format retrieved Document chunks into a single context string for the prompt.

    Each chunk is separated by a clear delimiter and prefixed with its source
    information so the LLM can cite sources accurately.

    Args:
        docs: List of retrieved Document objects from the vector store.

    Returns:
        Formatted multi-line string combining all chunk texts with headers.
    """
    formatted_chunks = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "?")
        chunk_id = doc.metadata.get("chunk_id", f"chunk-{i}")

        header = f"[Source {i}: {source} | Page {page}]"
        formatted_chunks.append(f"{header}\n{doc.page_content}")

    return "\n\n" + "\n\n---\n\n".join(formatted_chunks)


def get_qa_chain(retriever: VectorStoreRetriever):
    """
    Build and return the RAG chain using LangChain Expression Language (LCEL).

    LCEL (LangChain Expression Language) uses a pipe (|) syntax to compose
    components into a declarative, streaming-compatible pipeline.

    Chain structure:
        RunnableParallel(context, question)
            → ChatPromptTemplate
            → ChatGoogleGenerativeAI (Gemini)
            → StrOutputParser

    Args:
        retriever: A VectorStoreRetriever from vectorstore.get_retriever()

    Returns:
        A callable LCEL chain. Invoke with:
            result = chain.invoke({"question": "What is attention?"})
            # result["answer"]           → the generated answer string
            # result["source_documents"] → list of retrieved Document objects

    Why LCEL over RetrievalQA?
        LCEL is LangChain's modern API. It is:
        - Streaming-compatible (answer tokens arrive in real-time)
        - Easier to inspect and debug mid-chain
        - More composable for the multi-agent architecture in Phase 2
        - The legacy RetrievalQA class is deprecated in LangChain 0.2+
    """
    if not GOOGLE_API_KEY:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. "
            "Copy .env.example to .env and add your Gemini API key."
        )

    # -----------------------------------------------------------------------
    # Initialise Gemini LLM via LangChain's Google GenAI integration
    # -----------------------------------------------------------------------
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_TOKENS,
        # convert_system_message_to_human=True is required for some Gemini versions
        # that don't natively support the "system" role in the chat API.
        convert_system_message_to_human=True,
    )

    # -----------------------------------------------------------------------
    # Build the prompt template
    # -----------------------------------------------------------------------
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", RAG_HUMAN_PROMPT),
    ])

    # -----------------------------------------------------------------------
    # Compose the LCEL chain
    # -----------------------------------------------------------------------
    # Step 1: Run retriever and pass question through in parallel
    #   - "context" branch: retrieve chunks → format as string for prompt
    #   - "question" branch: pass the raw question through unchanged
    #   - "source_documents" branch: keep raw Document objects for attribution UI
    retrieve_and_format = RunnableParallel(
        context=retriever | _format_docs,
        question=RunnablePassthrough(),
        source_documents=retriever,   # Raw docs returned alongside the answer
    )

    # Step 2: Build the final answer chain (context + question → prompt → LLM → string)
    answer_chain = (
        {
            "context": lambda x: x["context"],
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # -----------------------------------------------------------------------
    # Wrap into a single callable that returns both answer AND source docs
    # -----------------------------------------------------------------------
    def run_chain(query: str) -> dict:
        """
        Execute the full RAG pipeline for a user query.

        Args:
            query: The user's natural language question.

        Returns:
            dict with keys:
                "answer"           : str — the LLM's grounded answer
                "source_documents" : list[Document] — retrieved chunks for attribution
        """
        # Retrieve documents and format context
        retrieved = retrieve_and_format.invoke(query)

        # Generate the answer from the retrieved context
        answer = answer_chain.invoke(retrieved)

        return {
            "answer": answer,
            "source_documents": retrieved["source_documents"],
        }

    print(f"[QA Chain] ✓ Chain ready (model={GEMINI_MODEL}, temp={LLM_TEMPERATURE})")
    return run_chain
