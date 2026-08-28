"""
rag_pipeline.py

Ties together retrieval (FAISS similarity search) and generation (Groq
LLM) into a single RAG pipeline. Given a user question, a vector store,
and an LLM, it retrieves relevant chunks, builds a grounded prompt, calls
the LLM, and returns a structured answer including sources and the type
of knowledge used.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage

from backend.prompts import SYSTEM_PROMPT, build_rag_prompt
from backend.vector_store import VectorStoreError, search_similar_documents

# FAISS returns L2 distance on normalized embeddings (range ~0 to 2).
# A lower score means higher similarity. Chunks scoring above this
# threshold are treated as not relevant enough to confidently ground an
# answer in the documents.
RELEVANCE_SCORE_THRESHOLD = 1.0

MAX_HISTORY_TURNS = 4  # number of recent user/assistant exchanges to include


class RAGPipelineError(Exception):
    """Raised when the RAG pipeline fails to produce an answer."""


def _format_history(chat_history: Optional[List[Dict[str, str]]]) -> str:
    """Format the most recent chat turns into a short text block."""
    if not chat_history:
        return ""

    recent = chat_history[-(MAX_HISTORY_TURNS * 2):]
    lines = []
    for turn in recent:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = turn.get("content", "")
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _build_context(relevant_docs: List[tuple]) -> str:
    """Concatenate retrieved chunk contents into a single context block."""
    context_parts = []
    for doc, _score in relevant_docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")
        context_parts.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(context_parts)


def answer_question(
    question: str,
    vector_store: Optional[FAISS],
    llm: Any,
    k: int = 4,
    chat_history: Optional[List[Dict[str, str]]] = None,
    relevance_threshold: float = RELEVANCE_SCORE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Execute the full RAG pipeline for a single user question.

    Args:
        question: The user's question.
        vector_store: The FAISS vector store to search (may be None if no
            knowledge base has been built yet).
        llm: A configured LangChain chat model (e.g. from backend.llm.get_llm).
        k: Number of top chunks to retrieve.
        chat_history: Optional list of prior turns, each a dict with
            "role" ("user"/"assistant") and "content".
        relevance_threshold: Max FAISS L2 distance for a chunk to be
            considered relevant.

    Returns:
        A dict with keys: "answer", "sources", "knowledge_type".

    Raises:
        RAGPipelineError: If the question is empty or the LLM call fails.
    """
    if not question or not question.strip():
        raise RAGPipelineError("Please enter a question before submitting.")

    relevant_docs: List[tuple] = []
    if vector_store is not None:
        try:
            results = search_similar_documents(vector_store, question, k=k)
            relevant_docs = [
                (doc, score) for doc, score in results if score <= relevance_threshold
            ]
        except VectorStoreError as exc:
            raise RAGPipelineError(str(exc)) from exc

    has_document_context = len(relevant_docs) > 0
    context_text = _build_context(relevant_docs) if has_document_context else ""
    history_text = _format_history(chat_history)

    prompt = build_rag_prompt(context=context_text, question=question, history=history_text)

    try:
        response = llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        answer_text = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:  # noqa: BLE001
        raise RAGPipelineError(
            "The LLM request failed. This may be due to an invalid API key, "
            "a rate limit, or a network issue. Please try again."
        ) from exc

    if not answer_text or not answer_text.strip():
        raise RAGPipelineError("The LLM returned an empty response. Please try again.")

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "unknown"),
            "content": doc.page_content,
            "score": round(float(score), 4),
        }
        for doc, score in relevant_docs
    ]

    if has_document_context and _looks_like_general_knowledge(answer_text):
        knowledge_type = "document + general"
    elif has_document_context:
        knowledge_type = "document"
    else:
        knowledge_type = "general"

    return {
        "answer": answer_text.strip(),
        "sources": sources,
        "knowledge_type": knowledge_type,
    }


def _looks_like_general_knowledge(answer_text: str) -> bool:
    """
    Lightweight heuristic to detect whether the model's own answer signals
    that it supplemented document context with general knowledge, based on
    common phrasing the system prompt asks it to use.
    """
    markers = [
        "general knowledge",
        "not mentioned in the document",
        "not found in the document",
        "based on my general",
        "outside the uploaded documents",
    ]
    lowered = answer_text.lower()
    return any(marker in lowered for marker in markers)
