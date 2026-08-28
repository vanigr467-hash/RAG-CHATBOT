"""
document_processor.py

Handles loading of uploaded PDF files, text extraction, cleaning, and
splitting into chunks suitable for embedding. Each resulting chunk retains
metadata (source filename and page number) so that answers can later be
cited back to their origin document.
"""

from __future__ import annotations

import os
import re
import tempfile
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be read or processed."""


def load_pdf(file_bytes: bytes, filename: str) -> List[Document]:
    """
    Load a single PDF from raw bytes and return a list of LangChain
    Document objects, one per page, with metadata attached.

    Args:
        file_bytes: Raw bytes of the uploaded PDF file.
        filename: Original filename of the uploaded PDF (used for metadata
            and error messages).

    Returns:
        A list of Document objects, one per non-empty page.

    Raises:
        PDFProcessingError: If the file cannot be parsed as a PDF or
            contains no extractable text.
    """
    if not file_bytes:
        raise PDFProcessingError(f"'{filename}' is empty and cannot be processed.")

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_path)
            raw_pages = loader.load()
        except Exception as exc:  # noqa: BLE001 - surfaced as a clean error
            raise PDFProcessingError(
                f"'{filename}' appears to be corrupted or is not a valid PDF."
            ) from exc

        if not raw_pages:
            raise PDFProcessingError(f"No pages could be read from '{filename}'.")

        pages: List[Document] = []
        for page in raw_pages:
            page_number = page.metadata.get("page", 0) + 1  # 1-indexed for display
            page.metadata = {"source": filename, "page": page_number}
            pages.append(page)

        return pages
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def clean_text(text: str) -> str:
    """
    Normalize whitespace and strip artifacts commonly found in extracted
    PDF text (repeated blank lines, stray form-feed characters, etc.).

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text with normalized whitespace.
    """
    if not text:
        return ""

    text = text.replace("\x0c", " ")  # form feed characters
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def extract_text(documents: List[Document]) -> List[Document]:
    """
    Apply text cleaning to a list of page-level Document objects and drop
    pages that contain no meaningful text after cleaning.

    Args:
        documents: List of raw page Documents (as returned by load_pdf).

    Returns:
        List of Documents with cleaned page_content, excluding empty pages.
    """
    cleaned_documents: List[Document] = []
    for doc in documents:
        cleaned = clean_text(doc.page_content)
        if cleaned:
            doc.page_content = cleaned
            cleaned_documents.append(doc)
    return cleaned_documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Split cleaned page-level Documents into smaller overlapping chunks
    using a recursive character splitter, preserving source/page metadata
    on every resulting chunk.

    Args:
        documents: List of cleaned page Documents.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of chunked Document objects.
    """
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Ensure metadata survives the split (LangChain propagates it, but we
    # defend against any missing keys explicitly).
    for chunk in chunks:
        chunk.metadata.setdefault("source", "unknown")
        chunk.metadata.setdefault("page", "unknown")

    return chunks


def process_uploaded_pdfs(
    uploaded_files,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Full pipeline: load, clean, and chunk a list of Streamlit
    UploadedFile objects.

    Args:
        uploaded_files: Iterable of Streamlit UploadedFile objects.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of chunked Document objects ready for embedding.

    Raises:
        PDFProcessingError: If a file fails to process. The filename is
            included in the error message so the caller can display it.
    """
    all_chunks: List[Document] = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        pages = load_pdf(file_bytes, uploaded_file.name)
        cleaned_pages = extract_text(pages)

        if not cleaned_pages:
            raise PDFProcessingError(
                f"'{uploaded_file.name}' contains no extractable text "
                "(it may be a scanned/image-only PDF)."
            )

        chunks = split_documents(cleaned_pages, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)

    return all_chunks
