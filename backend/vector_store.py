"""
vector_store.py

Wraps FAISS vector store creation, persistence, loading, and similarity
search. Document chunks are embedded using the shared embedding model and
indexed for fast semantic retrieval.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStoreError(Exception):
    """Raised when the FAISS vector store cannot be built, saved, or loaded."""


def create_vector_store(
    chunks: List[Document],
    embedding_model: HuggingFaceEmbeddings,
) -> FAISS:
    """
    Build a new FAISS vector store from document chunks.

    Args:
        chunks: List of chunked Document objects to index.
        embedding_model: The embedding model used to vectorize chunks.

    Returns:
        A populated FAISS vector store.

    Raises:
        VectorStoreError: If no chunks are provided or indexing fails.
    """
    if not chunks:
        raise VectorStoreError("No document chunks were provided to index.")

    try:
        return FAISS.from_documents(chunks, embedding_model)
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(f"Failed to build the FAISS index: {exc}") from exc


def save_vector_store(vector_store: FAISS, path: str) -> None:
    """
    Persist a FAISS vector store to disk.

    Args:
        vector_store: The FAISS index to save.
        path: Directory path to save the index into.

    Raises:
        VectorStoreError: If saving fails.
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        vector_store.save_local(path)
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(f"Failed to save the FAISS index: {exc}") from exc


def load_vector_store(path: str, embedding_model: HuggingFaceEmbeddings) -> FAISS:
    """
    Load a previously saved FAISS vector store from disk.

    Note: FAISS's local deserialization uses pickle under the hood. This
    function only ever loads indexes that this application itself wrote to
    `path` in the current session/environment — never load a FAISS index
    directory sourced from an untrusted or external location.

    Args:
        path: Directory path where the index was saved.
        embedding_model: The embedding model matching the one used to
            build the index (must be identical for correct results).

    Returns:
        The loaded FAISS vector store.

    Raises:
        VectorStoreError: If the index does not exist or fails to load.
    """
    if not Path(path).exists():
        raise VectorStoreError(f"No FAISS index found at '{path}'.")

    try:
        return FAISS.load_local(
            path,
            embedding_model,
            allow_dangerous_deserialization=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(f"Failed to load the FAISS index: {exc}") from exc


def search_similar_documents(
    vector_store: FAISS,
    query: str,
    k: int = 4,
) -> List[Tuple[Document, float]]:
    """
    Perform a similarity search against the FAISS index.

    Args:
        vector_store: The FAISS vector store to search.
        query: The user's question or search text.
        k: Number of top matching chunks to retrieve.

    Returns:
        A list of (Document, similarity_score) tuples, ordered by
        relevance (lower score = more similar, since FAISS returns L2
        distance by default).

    Raises:
        VectorStoreError: If the search fails.
    """
    if not query or not query.strip():
        raise VectorStoreError("Cannot search with an empty query.")

    try:
        return vector_store.similarity_search_with_score(query, k=k)
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(f"Similarity search failed: {exc}") from exc


def reset_vector_store_directory(path: str) -> None:
    """
    Delete a saved FAISS index directory entirely, used when resetting the
    knowledge base.

    Args:
        path: Directory path where the index was saved.
    """
    dir_path = Path(path)
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
