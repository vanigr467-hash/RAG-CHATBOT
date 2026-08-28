"""
embeddings.py

Loads and caches a Hugging Face sentence-transformer embedding model used
to convert both document chunks and user questions into vectors for FAISS
similarity search.
"""

from __future__ import annotations

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> HuggingFaceEmbeddings:
    """
    Load (and cache across Streamlit reruns) a Hugging Face sentence
    embedding model.

    Args:
        model_name: Hugging Face model identifier. Defaults to a
            lightweight model well-suited for local development.

    Returns:
        A HuggingFaceEmbeddings instance ready to embed text.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
